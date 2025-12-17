use anyhow::Result;
use base64::{engine::general_purpose::STANDARD, Engine};
use hmac::{Hmac, Mac};
use sha2::{Digest, Sha256, Sha512};

type HmacSha512 = Hmac<Sha512>;

pub struct KrakenAuth {
    api_key: String,
    api_secret: Vec<u8>,
}

impl KrakenAuth {
    pub fn new(api_key: &str, api_secret: &str) -> Result<Self> {
        let decoded_secret = STANDARD.decode(api_secret)?;
        Ok(Self {
            api_key: api_key.to_string(),
            api_secret: decoded_secret,
        })
    }

    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Sign a REST API request
    /// Returns the signature for the API-Sign header
    pub fn sign_request(&self, uri_path: &str, nonce: u64, post_data: &str) -> Result<String> {
        // SHA256(nonce + post_data)
        let mut sha256 = Sha256::new();
        sha256.update(nonce.to_string());
        sha256.update(post_data);
        let sha256_result = sha256.finalize();

        // HMAC-SHA512(uri_path + SHA256_result, api_secret)
        let mut hmac = HmacSha512::new_from_slice(&self.api_secret)?;
        hmac.update(uri_path.as_bytes());
        hmac.update(&sha256_result);
        let hmac_result = hmac.finalize();

        Ok(STANDARD.encode(hmac_result.into_bytes()))
    }

    /// Generate a nonce for API requests
    pub fn generate_nonce() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nonce_generation() {
        let nonce1 = KrakenAuth::generate_nonce();
        std::thread::sleep(std::time::Duration::from_millis(1));
        let nonce2 = KrakenAuth::generate_nonce();
        assert!(nonce2 > nonce1);
    }
}
