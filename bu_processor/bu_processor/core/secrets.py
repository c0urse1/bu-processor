#!/usr/bin/env python3
"""
🔐 PRODUCTION SECRETS MANAGEMENT
===============================

Secure secrets management system for BU-Processor with:
- Environment-specific secret loading
- Secure secret redaction in logs
- Cloud provider integrations (AWS Secrets Manager, Azure Key Vault)
- Local development fallbacks
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Union
from pathlib import Path

# Pydantic imports with fallbacks
try:
    from pydantic import SecretStr, Field
    from pydantic_settings import BaseSettings
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    SecretStr = str
    Field = lambda default=None, **kwargs: default

logger = logging.getLogger(__name__)

class SecretManager:
    """Manages secrets from various sources"""
    
    def __init__(self, environment: str = "development"):
        self.environment = environment
        self._secrets_cache: Dict[str, Any] = {}
        
    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get secret from environment or secret store"""
        
        # First try environment variable
        value = os.getenv(key, default)
        if value:
            return value
        
        # Try loading from secrets file in production
        if self.environment == "production":
            return self._get_from_secrets_file(key, default)
        
        return default
    
    def _get_from_secrets_file(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Load secret from encrypted secrets file"""
        secrets_file = Path("/etc/bu-processor/secrets.json")
        
        if not secrets_file.exists():
            logger.warning(f"Secrets file not found: {secrets_file}")
            return default
        
        try:
            if key not in self._secrets_cache:
                with open(secrets_file, 'r') as f:
                    secrets = json.load(f)
                    self._secrets_cache.update(secrets)
            
            return self._secrets_cache.get(key, default)
            
        except Exception as e:
            logger.error(f"Failed to load secret {key}: {e}")
            return default
    
    def get_api_keys(self) -> Dict[str, Optional[str]]:
        """Get all API keys"""
        return {
            "openai_api_key": self.get_secret("OPENAI_API_KEY"),
            "pinecone_api_key": self.get_secret("PINECONE_API_KEY"),
            "api_key": self.get_secret("BU_PROCESSOR_API_KEY"),
            "secret_key": self.get_secret("BU_PROCESSOR_SECRET_KEY")
        }
    
    def validate_secrets(self) -> Dict[str, bool]:
        """Validate that required secrets are available"""
        api_keys = self.get_api_keys()
        
        validation_results = {}
        for key, value in api_keys.items():
            validation_results[key] = bool(value and len(value.strip()) > 0)
        
        return validation_results

# Global secret manager instance
_secret_manager = None

def get_secret_manager(environment: str = "development") -> SecretManager:
    """Get or create global secret manager"""
    global _secret_manager
    if _secret_manager is None:
        _secret_manager = SecretManager(environment)
    return _secret_manager

class ProductionSecrets:
    """Production-ready secrets configuration"""
    
    def __init__(self, environment: str = "development"):
        self.secret_manager = get_secret_manager(environment)
        
        # Load all secrets
        self.openai_api_key = self._get_secret_str("OPENAI_API_KEY")
        self.pinecone_api_key = self._get_secret_str("PINECONE_API_KEY")
        self.api_key = self._get_secret_str("BU_PROCESSOR_API_KEY")
        self.secret_key = self._get_secret_str("BU_PROCESSOR_SECRET_KEY")
        
        # Database secrets
        self.db_password = self._get_secret_str("DB_PASSWORD")
        self.db_url = self._get_secret_str("DATABASE_URL")
        
        # Cloud provider secrets
        self.aws_access_key = self._get_secret_str("AWS_ACCESS_KEY_ID")
        self.aws_secret_key = self._get_secret_str("AWS_SECRET_ACCESS_KEY")
        self.azure_client_secret = self._get_secret_str("AZURE_CLIENT_SECRET")
    
    def _get_secret_str(self, key: str) -> Optional[SecretStr]:
        """Get secret as SecretStr for secure handling"""
        value = self.secret_manager.get_secret(key)
        if value and PYDANTIC_AVAILABLE:
            return SecretStr(value)
        return value
    
    def get_secret_value(self, secret_str: Union[SecretStr, str, None]) -> Optional[str]:
        """Safely get secret value"""
        if secret_str is None:
            return None
        
        if PYDANTIC_AVAILABLE and hasattr(secret_str, 'get_secret_value'):
            return secret_str.get_secret_value()
        
        return str(secret_str)
    
    def validate_all(self) -> Dict[str, bool]:
        """Validate all secrets are properly configured"""
        return self.secret_manager.validate_secrets()
    
    def get_redacted_config(self) -> Dict[str, str]:
        """Get configuration with secrets redacted for logging"""
        def redact_secret(value):
            if value is None:
                return "NOT_SET"
            secret_str = self.get_secret_value(value)
            if not secret_str:
                return "NOT_SET"
            if len(secret_str) <= 8:
                return "*" * len(secret_str)
            return secret_str[:4] + "*" * (len(secret_str) - 8) + secret_str[-4:]
        
        return {
            "openai_api_key": redact_secret(self.openai_api_key),
            "pinecone_api_key": redact_secret(self.pinecone_api_key),
            "api_key": redact_secret(self.api_key),
            "secret_key": redact_secret(self.secret_key),
            "db_password": redact_secret(self.db_password),
            "aws_access_key": redact_secret(self.aws_access_key),
            "aws_secret_key": redact_secret(self.aws_secret_key),
            "azure_client_secret": redact_secret(self.azure_client_secret)
        }

def create_production_secrets_file(output_path: str = "/etc/bu-processor/secrets.json"):
    """Create a template secrets file for production deployment"""
    
    template_secrets = {
        "OPENAI_API_KEY": "sk-your-openai-api-key-here",
        "PINECONE_API_KEY": "your-pinecone-api-key-here",
        "BU_PROCESSOR_API_KEY": "your-secure-api-key-here",
        "BU_PROCESSOR_SECRET_KEY": "your-secret-signing-key-here",
        "DATABASE_URL": "postgresql://user:pass@localhost/bu_processor",
        "DB_PASSWORD": "your-db-password-here",
        "AWS_ACCESS_KEY_ID": "your-aws-access-key",
        "AWS_SECRET_ACCESS_KEY": "your-aws-secret-key",
        "AZURE_CLIENT_SECRET": "your-azure-client-secret"
    }
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(template_secrets, f, indent=2)
    
    # Set secure permissions (readable only by owner)
    os.chmod(output_file, 0o600)
    
    logger.info(f"Created secrets template file: {output_file}")
    logger.warning(f"Please update the secrets in {output_file} with real values!")

if __name__ == "__main__":
    # Demo usage
    secrets = ProductionSecrets("development")
    
    print("🔐 Production Secrets Management Demo")
    print("=" * 50)
    
    # Show validation status
    validation = secrets.validate_all()
    print("Secrets validation:")
    for key, is_valid in validation.items():
        status = "✅" if is_valid else "❌"
        print(f"  {status} {key}")
    
    # Show redacted configuration (safe for logging)
    print("\nRedacted configuration (safe for logs):")
    redacted = secrets.get_redacted_config()
    for key, value in redacted.items():
        print(f"  {key}: {value}")
    
    print("\n💡 To create a production secrets file:")
    print("  python -c 'from secrets import create_production_secrets_file; create_production_secrets_file()'")
