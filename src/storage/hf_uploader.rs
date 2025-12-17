//! HuggingFace Dataset Uploader
//!
//! Uploads Parquet files to HuggingFace Hub using Python's huggingface_hub library.
//! This approach handles LFS (Large File Storage) automatically.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};
use std::process::Command;
use tracing::{debug, error, info};

/// Response from the Python upload script
#[derive(Debug, Deserialize)]
struct UploadResult {
    success: bool,
    #[serde(default)]
    files_uploaded: usize,
    #[serde(default)]
    bytes_uploaded: u64,
    #[serde(default)]
    error: Option<String>,
}

/// HuggingFace Dataset uploader
pub struct HfUploader {
    repo_id: String,
    token: String,
    data_dir: PathBuf,
    script_path: PathBuf,
    total_files_uploaded: usize,
    total_bytes_uploaded: u64,
}

impl HfUploader {
    /// Create a new HfUploader
    pub fn new(
        repo_id: &str,
        data_dir: &Path,
        token: &str,
        _upload_delay_hours: u32, // Not used with Python uploader
    ) -> Result<Self> {
        // Find the Python script
        let script_path = Self::find_script()?;

        info!(
            "HuggingFace uploader initialized for repo: {} (using Python)",
            repo_id
        );

        Ok(Self {
            repo_id: repo_id.to_string(),
            token: token.to_string(),
            data_dir: data_dir.to_path_buf(),
            script_path,
            total_files_uploaded: 0,
            total_bytes_uploaded: 0,
        })
    }

    /// Find the Python upload script
    fn find_script() -> Result<PathBuf> {
        // Try common locations
        let candidates = [
            PathBuf::from("scripts/hf_upload.py"),
            PathBuf::from("./scripts/hf_upload.py"),
        ];

        for path in &candidates {
            if path.exists() {
                debug!("Found HF upload script at: {:?}", path);
                return Ok(path.clone());
            }
        }

        // Try relative to executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let script = exe_dir.join("scripts/hf_upload.py");
                if script.exists() {
                    return Ok(script);
                }
            }
        }

        anyhow::bail!(
            "HuggingFace upload script not found. Expected at: scripts/hf_upload.py"
        )
    }

    /// Sync data to HuggingFace using Python script
    pub async fn sync(&mut self) -> Result<usize> {
        info!("Starting HuggingFace sync via Python...");

        // Run Python script
        let output = Command::new("python3")
            .arg(&self.script_path)
            .arg("--repo-id")
            .arg(&self.repo_id)
            .arg("--data-dir")
            .arg(&self.data_dir)
            .arg("--token")
            .arg(&self.token)
            .arg("--json")
            .output()
            .context("Failed to execute Python upload script")?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        if !stderr.is_empty() {
            debug!("Python stderr: {}", stderr);
        }

        // Parse JSON result
        let result: UploadResult = serde_json::from_str(&stdout).with_context(|| {
            format!(
                "Failed to parse Python output. stdout: {}, stderr: {}",
                stdout, stderr
            )
        })?;

        if result.success {
            if result.files_uploaded > 0 {
                info!(
                    "Uploaded {} files ({:.2} MB) to HuggingFace",
                    result.files_uploaded,
                    result.bytes_uploaded as f64 / 1_000_000.0
                );
                self.total_files_uploaded += result.files_uploaded;
                self.total_bytes_uploaded += result.bytes_uploaded;
            } else {
                debug!("No new files to upload");
            }
            Ok(result.files_uploaded)
        } else {
            let error_msg = result.error.unwrap_or_else(|| "Unknown error".to_string());
            error!("HuggingFace upload failed: {}", error_msg);
            anyhow::bail!("Upload failed: {}", error_msg)
        }
    }

    /// Get upload statistics
    pub fn stats(&self) -> (usize, u64) {
        (self.total_files_uploaded, self.total_bytes_uploaded)
    }

    /// Check if Python and huggingface_hub are available
    pub fn check_dependencies() -> Result<()> {
        let output = Command::new("python3")
            .arg("-c")
            .arg("import huggingface_hub; print(huggingface_hub.__version__)")
            .output()
            .context("Failed to check Python dependencies")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("No module named") {
                anyhow::bail!(
                    "huggingface_hub not installed. Run: pip install huggingface_hub"
                );
            }
            anyhow::bail!("Python check failed: {}", stderr);
        }

        let version = String::from_utf8_lossy(&output.stdout);
        info!("huggingface_hub version: {}", version.trim());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upload_result_parsing() {
        let json = r#"{"success": true, "files_uploaded": 5, "bytes_uploaded": 1024}"#;
        let result: UploadResult = serde_json::from_str(json).unwrap();
        assert!(result.success);
        assert_eq!(result.files_uploaded, 5);
        assert_eq!(result.bytes_uploaded, 1024);
    }

    #[test]
    fn test_upload_result_error() {
        let json = r#"{"success": false, "error": "Test error"}"#;
        let result: UploadResult = serde_json::from_str(json).unwrap();
        assert!(!result.success);
        assert_eq!(result.error, Some("Test error".to_string()));
    }
}
