#!/usr/bin/env python3
"""
Test script for the MCP Image Converter Server
"""

import json
import sys
from pathlib import Path

def test_dependencies():
    """Test if all required dependencies are available"""
    print("Testing dependencies...")
    
    required_packages = [
        "numpy",
        "tifffile", 
        "imageio",
        "zarr",
        "mcp"
    ]
    
    optional_packages = [
        "ome_zarr",
        "napari"
    ]
    
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (required)")
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"⚠ {package} (optional)")
            missing_optional.append(package)
    
    if missing_required:
        print(f"\nMissing required packages: {missing_required}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\nMissing optional packages: {missing_optional}")
        print("For full functionality, install with: pip install 'napari[all]'")
    
    return True

def test_image_converter():
    """Test the ImageConverter class directly"""
    print("\nTesting ImageConverter class...")
    
    try:
        # Import the ImageConverter class
        from server import ImageConverter
        
        # Test with a sample image (if available)
        test_image_path = "/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff"
        
        if Path(test_image_path).exists():
            print(f"Testing with image: {test_image_path}")
            
            # Test loading image
            image, metadata = ImageConverter.load_image(test_image_path)
            print(f"✓ Successfully loaded image")
            print(f"  - Shape: {metadata['shape']}")
            print(f"  - Dtype: {metadata['dtype']}")
            print(f"  - File size: {metadata['file_size_mb']} MB")
            
            # Test simple zarr conversion
            output_path = "/tmp/test_output.zarr"
            result = ImageConverter.save_simple_zarr(image, output_path)
            print(f"✓ Successfully converted to simple zarr: {result['output_path']}")
            
            # Clean up
            import shutil
            if Path(output_path).exists():
                shutil.rmtree(output_path)
            
            return True
        else:
            print(f"⚠ Test image not found: {test_image_path}")
            print("Skipping image processing tests")
            return True
            
    except Exception as e:
        print(f"✗ Error testing ImageConverter: {e}")
        return False

def test_mcp_server_import():
    """Test that the MCP server can be imported"""
    print("\nTesting MCP server import...")
    
    try:
        import server
        print("✓ MCP server module imported successfully")
        
        # Check if required components exist
        if hasattr(server, 'server'):
            print("✓ MCP server instance found")
        else:
            print("✗ MCP server instance not found")
            return False
            
        if hasattr(server, 'ImageConverter'):
            print("✓ ImageConverter class found")
        else:
            print("✗ ImageConverter class not found")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error importing MCP server: {e}")
        return False

def main():
    """Main test function"""
    print("=== MCP Image Converter Server Test ===\n")
    
    # Test dependencies first
    if not test_dependencies():
        return
    
    # Test MCP server import
    if not test_mcp_server_import():
        return
    
    # Test image converter functionality
    if not test_image_converter():
        return
    
    print("\n🎉 All tests passed! The MCP server is ready to use.")
    print("\nTo run the server:")
    print("  python server.py")
    print("\nTo test with a client, use the MCP protocol over stdin/stdout.")

if __name__ == "__main__":
    main() 