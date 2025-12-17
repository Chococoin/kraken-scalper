//! HuggingFace Dataset Uploader
//!
//! Uploads Parquet files to HuggingFace Hub for ML training datasets.

use anyhow::{Context, Result};
use base64::Engine;
use chrono::{DateTime, Timelike, Utc};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use tracing::{debug, error, info, warn};

const HF_API_BASE: &str = "https://huggingface.co/api";

/// State tracking for uploaded files
#[derive(Debug, Serialize, Deserialize, Default)]
pub struct UploadState {
    pub last_sync: Option<DateTime<Utc>>,
    pub uploaded_files: HashMap<String, FileUploadRecord>,
}

/// Record of an uploaded file
#[derive(Debug, Serialize, Deserialize)]
pub struct FileUploadRecord {
    pub hash: String,
    pub uploaded_at: DateTime<Utc>,
    pub size_bytes: u64,
}

/// Response from HuggingFace commit API
#[derive(Debug, Deserialize)]
struct CommitResponse {
    #[serde(rename = "commitOid")]
    commit_oid: Option<String>,
}

/// HuggingFace Dataset uploader
pub struct HfUploader {
    client: Client,
    repo_id: String,
    token: String,
    data_dir: PathBuf,
    state_path: PathBuf,
    state: UploadState,
    upload_delay_hours: u32,
}

impl HfUploader {
    /// Create a new HfUploader
    pub fn new(
        repo_id: &str,
        data_dir: &Path,
        token: &str,
        upload_delay_hours: u32,
    ) -> Result<Self> {
        let state_path = data_dir.join(".hf_upload_state.json");
        let state = Self::load_state(&state_path).unwrap_or_default();

        info!(
            "HuggingFace uploader initialized for repo: {}, {} files tracked",
            repo_id,
            state.uploaded_files.len()
        );

        Ok(Self {
            client: Client::new(),
            repo_id: repo_id.to_string(),
            token: token.to_string(),
            data_dir: data_dir.to_path_buf(),
            state_path,
            state,
            upload_delay_hours,
        })
    }

    /// Find files eligible for upload (completed hours only)
    pub fn find_uploadable_files(&self) -> Result<Vec<PathBuf>> {
        let cutoff = Utc::now() - chrono::Duration::hours(self.upload_delay_hours as i64);
        let cutoff_hour = cutoff.hour();
        let cutoff_date = cutoff.format("%Y-%m-%d").to_string();

        let mut files = Vec::new();

        for category in ["crypto", "stocks"] {
            for data_type in ["ticker", "book", "ohlc", "trade"] {
                let type_dir = self.data_dir.join(category).join(data_type);
                if !type_dir.exists() {
                    continue;
                }

                let date_entries = match fs::read_dir(&type_dir) {
                    Ok(entries) => entries,
                    Err(_) => continue,
                };

                for date_entry in date_entries.flatten() {
                    let date_str = date_entry.file_name().to_string_lossy().to_string();

                    // Skip future dates
                    if date_str > cutoff_date {
                        continue;
                    }

                    let file_entries = match fs::read_dir(date_entry.path()) {
                        Ok(entries) => entries,
                        Err(_) => continue,
                    };

                    for file_entry in file_entries.flatten() {
                        let path = file_entry.path();

                        // Only process parquet files
                        if path.extension().map_or(true, |e| e != "parquet") {
                            continue;
                        }

                        // Get hour from filename (e.g., "13.parquet" -> 13)
                        let hour: u32 = path
                            .file_stem()
                            .and_then(|s| s.to_str())
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(24); // Invalid hour will be skipped

                        // Skip current hour on cutoff date (file might still be written)
                        if date_str == cutoff_date && hour >= cutoff_hour {
                            continue;
                        }

                        files.push(path);
                    }
                }
            }
        }

        debug!("Found {} uploadable files", files.len());
        Ok(files)
    }

    /// Sync pending files to HuggingFace
    pub async fn sync(&mut self) -> Result<usize> {
        let files = self.find_uploadable_files()?;
        let mut uploaded = 0;
        let mut to_upload = Vec::new();

        // Check which files need uploading
        for file_path in files {
            let rel_path = file_path
                .strip_prefix(&self.data_dir)
                .context("Failed to get relative path")?;
            let rel_str = rel_path.to_string_lossy().to_string();

            // Calculate hash
            let hash = self.hash_file(&file_path)?;

            // Check if already uploaded with same hash
            if let Some(record) = self.state.uploaded_files.get(&rel_str) {
                if record.hash == hash {
                    continue; // Already uploaded, skip
                }
                debug!("File modified, will re-upload: {}", rel_str);
            }

            to_upload.push((file_path, rel_str, hash));
        }

        if to_upload.is_empty() {
            debug!("No new files to upload");
            return Ok(0);
        }

        info!("Uploading {} files to HuggingFace...", to_upload.len());

        // Upload files (batch into commits of 10 files max to avoid API limits)
        for chunk in to_upload.chunks(10) {
            match self.upload_batch(chunk).await {
                Ok(count) => {
                    uploaded += count;

                    // Update state for uploaded files
                    for (file_path, rel_str, hash) in chunk {
                        let size = file_path.metadata().map(|m| m.len()).unwrap_or(0);
                        self.state.uploaded_files.insert(
                            rel_str.clone(),
                            FileUploadRecord {
                                hash: hash.clone(),
                                uploaded_at: Utc::now(),
                                size_bytes: size,
                            },
                        );
                    }
                }
                Err(e) => {
                    error!("Failed to upload batch: {}", e);
                }
            }
        }

        // Save state
        self.state.last_sync = Some(Utc::now());
        self.save_state()?;

        if uploaded > 0 {
            info!("Successfully uploaded {} files to HuggingFace", uploaded);
        }

        Ok(uploaded)
    }

    /// Upload a batch of files in a single commit
    async fn upload_batch(&self, files: &[(PathBuf, String, String)]) -> Result<usize> {
        let mut operations = Vec::new();

        for (file_path, rel_str, _hash) in files {
            // Read file content
            let mut file = File::open(file_path)
                .with_context(|| format!("Failed to open file: {:?}", file_path))?;
            let mut content = Vec::new();
            file.read_to_end(&mut content)?;

            // Encode as base64
            let content_b64 = base64::engine::general_purpose::STANDARD.encode(&content);

            operations.push(serde_json::json!({
                "path": format!("data/{}", rel_str),
                "content": content_b64,
                "encoding": "base64"
            }));

            debug!("Prepared file for upload: {}", rel_str);
        }

        // Build commit message
        let summary = if files.len() == 1 {
            format!("Add {}", files[0].1)
        } else {
            format!("Add {} data files", files.len())
        };

        // Create commit via HF API
        let url = format!("{}/datasets/{}/commit/main", HF_API_BASE, self.repo_id);

        let payload = serde_json::json!({
            "summary": summary,
            "files": operations
        });

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.token))
            .header("Content-Type", "application/json")
            .json(&payload)
            .send()
            .await
            .context("Failed to send commit request")?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response
                .text()
                .await
                .unwrap_or_else(|_| "Unknown error".to_string());
            anyhow::bail!("HuggingFace API error ({}): {}", status, error_text);
        }

        let commit_response: CommitResponse = response
            .json()
            .await
            .context("Failed to parse commit response")?;

        if let Some(oid) = commit_response.commit_oid {
            debug!("Commit created: {}", oid);
        }

        Ok(files.len())
    }

    /// Calculate SHA256 hash of a file
    fn hash_file(&self, path: &Path) -> Result<String> {
        let mut file = File::open(path).context("Failed to open file for hashing")?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(format!("sha256:{:x}", hasher.finalize()))
    }

    /// Load upload state from disk
    fn load_state(path: &Path) -> Result<UploadState> {
        let content = fs::read_to_string(path).context("Failed to read state file")?;
        serde_json::from_str(&content).context("Failed to parse state file")
    }

    /// Save upload state to disk
    fn save_state(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.state)?;
        fs::write(&self.state_path, content).context("Failed to write state file")?;
        Ok(())
    }

    /// Get upload statistics
    pub fn stats(&self) -> (usize, u64) {
        let count = self.state.uploaded_files.len();
        let size: u64 = self.state.uploaded_files.values().map(|r| r.size_bytes).sum();
        (count, size)
    }

    /// Force re-upload all files (clear state)
    pub fn reset_state(&mut self) -> Result<()> {
        warn!("Resetting HuggingFace upload state - all files will be re-uploaded");
        self.state = UploadState::default();
        self.save_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_upload_state_serialization() {
        let mut state = UploadState::default();
        state.uploaded_files.insert(
            "crypto/ticker/2025-12-17/13.parquet".to_string(),
            FileUploadRecord {
                hash: "sha256:abc123".to_string(),
                uploaded_at: Utc::now(),
                size_bytes: 1024,
            },
        );

        let json = serde_json::to_string(&state).unwrap();
        let parsed: UploadState = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.uploaded_files.len(), 1);
    }

    #[test]
    fn test_hash_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "hello world").unwrap();

        let uploader = HfUploader {
            client: Client::new(),
            repo_id: "test/repo".to_string(),
            token: "test".to_string(),
            data_dir: dir.path().to_path_buf(),
            state_path: dir.path().join(".hf_upload_state.json"),
            state: UploadState::default(),
            upload_delay_hours: 1,
        };

        let hash = uploader.hash_file(&file_path).unwrap();
        assert!(hash.starts_with("sha256:"));
    }
}
