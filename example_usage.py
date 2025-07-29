#!/usr/bin/env python3
"""
Example usage of the Image Converter MCP Server functionality
This script demonstrates how to use the ImageConverter class directly
"""

import os
from pathlib import Path
from server import ImageConverter

def main():
    """Example usage of the ImageConverter"""
    
    # Example image path (update this to your actual image path)
    input_image = "/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff"
    output_zarr = "/Users/sesan/Desktop/output_image.ome.zarr"
    simple_zarr = "/Users/sesan/Desktop/simple_image.zarr"
    
    print("=== Image Converter Example ===\n")
    
    # Check if the input image exists
    if not Path(input_image).exists():
        print(f"⚠ Input image not found: {input_image}")
        print("Please update the input_image path in this script to point to your image file.")
        return
    
    try:
        # Step 1: Load and analyze the image
        print("1. Loading image...")
        image, metadata = ImageConverter.load_image(input_image)
        
        print(f"✓ Image loaded successfully!")
        print(f"   - Shape: {metadata['shape']}")
        print(f"   - Data type: {metadata['dtype']}")
        print(f"   - Min/Max values: {metadata['min_value']}/{metadata['max_value']}")
        print(f"   - File size: {metadata['file_size_mb']} MB")
        
        # Step 2: Convert to OME-Zarr format
        print("\n2. Converting to OME-Zarr format...")
        ome_result = ImageConverter.tiff_to_omezarr(image, output_zarr, scale=False)
        
        print(f"✓ OME-Zarr conversion successful!")
        print(f"   - Method used: {ome_result['method']}")
        print(f"   - Output path: {ome_result['output_path']}")
        print(f"   - Original shape: {ome_result['original_shape']}")
        print(f"   - Zarr shape: {ome_result['zarr_shape']}")
        
        # Step 3: Convert to simple zarr format
        print("\n3. Converting to simple zarr format...")
        simple_result = ImageConverter.save_simple_zarr(image, simple_zarr)
        
        print(f"✓ Simple zarr conversion successful!")
        print(f"   - Output path: {simple_result['output_path']}")
        print(f"   - Shape: {simple_result['shape']}")
        
        # Step 4: Create matplotlib viewer
        print("\n4. Creating matplotlib viewer...")
        viewer_result = ImageConverter.launch_matplotlib_viewer(input_image, colormap="green")
        
        if viewer_result["status"] == "success":
            print(f"✓ Matplotlib viewer created successfully!")
            print(f"   - Colormap: {viewer_result['viewer_settings']['colormap']}")
            print(f"   - Contrast limits: {viewer_result['viewer_settings']['contrast_limits']}")
            print(f"   - Output image: {viewer_result['viewer_settings']['output_image']}")
        else:
            print(f"⚠ Matplotlib viewer issue: {viewer_result['message']}")
        
        # Summary
        print("\n=== Summary ===")
        print(f"✓ Input image: {Path(input_image).name}")
        print(f"✓ OME-Zarr output: {Path(output_zarr).name}")
        print(f"✓ Simple zarr output: {Path(simple_zarr).name}")
        print(f"✓ Matplotlib viewer: {'Created' if viewer_result['status'] == 'success' else 'Not available'}")
        
        print("\n🎉 All operations completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during processing: {e}")
        print("Check that all dependencies are installed:")
        print("  pip install -r requirements.txt")

def demonstrate_mcp_tools():
    """Demonstrate the MCP tools that would be available"""
    print("\n=== Available MCP Tools ===")
    
    tools = [
        {
            "name": "load_image",
            "description": "Load an image file and get metadata",
            "example": {
                "file_path": "/path/to/image.tiff"
            }
        },
        {
            "name": "convert_to_omezarr", 
            "description": "Convert a TIFF image to OME-Zarr format",
            "example": {
                "input_path": "/path/to/input.tiff",
                "output_path": "/path/to/output.ome.zarr",
                "scale": True
            }
        },
        {
            "name": "convert_to_simple_zarr",
            "description": "Convert an image to simple zarr format", 
            "example": {
                "input_path": "/path/to/input.tiff",
                "output_path": "/path/to/output.zarr"
            }
        },
        {
            "name": "launch_matplotlib",
            "description": "Create matplotlib viewer with an image",
            "example": {
                "image_path": "/path/to/image.tiff",
                "colormap": "green"
            }
        },
        {
            "name": "get_image_info",
            "description": "Get detailed information about an image file",
            "example": {
                "file_path": "/path/to/image.tiff"
            }
        },
        {
            "name": "process_fluorescence_image",
            "description": "Process a fluorescence image with optimal settings",
            "example": {
                "input_path": "/path/to/fluorescence.tiff",
                "output_zarr_path": "/path/to/output.ome.zarr",
                "colormap": "green"
            }
        }
    ]
    
    for i, tool in enumerate(tools, 1):
        print(f"\n{i}. {tool['name']}")
        print(f"   Description: {tool['description']}")
        print(f"   Example: {tool['example']}")

if __name__ == "__main__":
    # Show available MCP tools
    demonstrate_mcp_tools()
    
    # Run the example
    main() 