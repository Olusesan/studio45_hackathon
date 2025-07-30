#!/usr/bin/env python3
"""
Setup script for Clause Desktop MCP integration
"""

import os
import sys
import json
import shutil
from pathlib import Path

def get_clause_config_dir():
    """Get the Clause Desktop configuration directory for the current platform"""
    if sys.platform == "darwin":  # macOS
        return Path.home() / "Library" / "Application Support" / "Clause"
    elif sys.platform == "win32":  # Windows
        return Path(os.environ.get("APPDATA", "")) / "Clause"
    else:  # Linux
        return Path.home() / ".config" / "Clause"

def setup_clause_config():
    """Set up Clause Desktop configuration"""
    
    print("=== Clause Desktop MCP Setup ===\n")
    
    # Get Clause config directory
    clause_dir = get_clause_config_dir()
    
    if not clause_dir.exists():
        print(f"⚠ Clause Desktop config directory not found: {clause_dir}")
        print("Please install Clause Desktop first: https://clause.ai/")
        return False
    
    print(f"✓ Found Clause Desktop config directory: {clause_dir}")
    
    # Check if config file already exists
    config_file = clause_dir / "clause_config.json"
    
    if config_file.exists():
        print(f"⚠ Config file already exists: {config_file}")
        backup_file = clause_dir / "clause_config.json.backup"
        shutil.copy2(config_file, backup_file)
        print(f"✓ Created backup: {backup_file}")
    
    # Copy our config file
    source_config = Path("clause_config.json")
    if not source_config.exists():
        print("❌ clause_config.json not found in current directory")
        return False
    
    try:
        shutil.copy2(source_config, config_file)
        print(f"✓ Copied config to: {config_file}")
        
        # Verify the config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print("✓ Configuration verified:")
        for server_name, server_config in config.get("mcpServers", {}).items():
            print(f"  - {server_name}: {server_config['command']} {' '.join(server_config['args'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error copying config: {e}")
        return False

def verify_dependencies():
    """Verify that all required dependencies are installed"""
    
    print("\n=== Verifying Dependencies ===\n")
    
    required_packages = [
        "numpy",
        "tifffile", 
        "imageio",
        "zarr",
        "mcp"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("\n✓ All required dependencies are installed")
    return True

def test_server():
    """Test that the MCP server can be imported and run"""
    
    print("\n=== Testing MCP Server ===\n")
    
    try:
        # Test import
        import server
        print("✓ MCP server module imported")
        
        # Test ImageConverter class
        from server import ImageConverter
        print("✓ ImageConverter class imported")
        
        # Test server instance
        if hasattr(server, 'server'):
            print("✓ MCP server instance found")
        else:
            print("✗ MCP server instance not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing server: {e}")
        return False

def create_test_image():
    """Create a test image if the original doesn't exist"""
    
    test_image_path = Path("/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff")
    
    if test_image_path.exists():
        print(f"✓ Test image found: {test_image_path}")
        return True
    
    print(f"⚠ Test image not found: {test_image_path}")
    print("Creating a sample test image...")
    
    try:
        import numpy as np
        from tifffile import imwrite
        
        # Create a sample fluorescence-like image
        image = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
        
        # Add some structure to make it look like fluorescence data
        y, x = np.ogrid[:512, :512]
        center_y, center_x = 256, 256
        radius = 100
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        image[mask] = image[mask] * 2 + 10000
        
        # Save as TIFF
        imwrite(test_image_path, image)
        print(f"✓ Created test image: {test_image_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test image: {e}")
        return False

def main():
    """Main setup function"""
    
    print("Setting up Clause Desktop MCP integration...\n")
    
    # Step 1: Verify dependencies
    if not verify_dependencies():
        return
    
    # Step 2: Test server
    if not test_server():
        return
    
    # Step 3: Create test image if needed
    create_test_image()
    
    # Step 4: Setup Clause config
    if not setup_clause_config():
        return
    
    print("\n=== Setup Complete! ===")
    print("✓ Dependencies verified")
    print("✓ MCP server tested")
    print("✓ Clause Desktop configuration updated")
    print("\nNext steps:")
    print("1. Restart Clause Desktop")
    print("2. Check that the 'image-converter' server is connected")
    print("3. Test with commands like:")
    print("   - 'Analyze the fluorescence image on my desktop'")
    print("   - 'Convert this image to OME-Zarr format'")
    print("   - 'Show me the image in napari'")
    print("\nFor troubleshooting, run: python test_clause_integration.py")

if __name__ == "__main__":
    main() 