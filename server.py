#!/usr/bin/env python3
"""
MCP Server for Image Conversion and Visualization
Provides tools for converting TIFF images to OME-Zarr format, launching napari viewer, and cell counting
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import shutil

# Image processing imports
import numpy as np
import imageio.v3 as iio
from tifffile import imread
import zarr

# Cell counting imports
try:
    from skimage import filters, morphology, measure, segmentation
    from skimage.feature import peak_local_maxima  # type: ignore
    from scipy import ndimage
    CELL_COUNTING_AVAILABLE = True
except ImportError:
    CELL_COUNTING_AVAILABLE = False

# MCP imports
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import ServerCapabilities
import mcp.server.stdio
import mcp.types as types

# Initialize the MCP server
server = Server("image-converter")

class ImageConverter:
    """Image conversion utilities for MCP server"""
    
    @staticmethod
    def load_image(file_path: str) -> tuple[np.ndarray, dict]:
        """Load image and return array with metadata"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Image file not found: {file_path}")
            
            # Try tifffile first (better for scientific images)
            try:
                image = imread(file_path)
            except Exception:
                # Fallback to imageio
                image = iio.imread(file_path)
            
            metadata = {
                "shape": image.shape,
                "dtype": str(image.dtype),
                "min_value": int(image.min()),
                "max_value": int(image.max()),
                "file_size_mb": round(os.path.getsize(file_path) / (1024*1024), 2)
            }
            
            return image, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load image: {str(e)}")
    
    @staticmethod
    def tiff_to_omezarr_v1(tiff_path, zarr_dir, scale=False):
        """
        Convert TIFF to OME-Zarr format using older API (from serve2.py)
        """
        try:
            from ome_zarr.writer import write_image
            from ome_zarr.io import parse_url
            import zarr
            
            # Load image
            img = iio.imread(tiff_path)
            print(f"Original image shape: {img.shape}")
            print(f"Original image dtype: {img.dtype}")
            
            # Reshape for OME-Zarr (expects CZYX format)
            if img.ndim == 2:
                img = img[None, None, ...]  # (1, 1, Y, X)
            elif img.ndim == 3:
                img = img[None, ...]  # (1, Z, Y, X)
            
            print(f"Reshaped image for OME-Zarr: {img.shape}")
            
            # Remove existing zarr directory if it exists
            if os.path.exists(zarr_dir):
                import shutil
                shutil.rmtree(zarr_dir)
            
            # Method 1: Try newer API
            try:
                # Use zarr.open_group directly instead of parse_url
                group = zarr.open_group(zarr_dir, mode='w')
                
                # Prepare scale factors properly
                scale_factors = [{"x": 2, "y": 2, "z": 2}] if scale else []
                
                write_image(
                    img, 
                    group=group, 
                    axes="czyx", 
                    scale_factors=scale_factors
                )
                print(f"Saved OME-Zarr to: {zarr_dir} (newer API)")
                return img
                
            except Exception as e1:
                print(f"Newer API failed: {e1}")
                
                # Method 2: Try direct zarr approach
                try:
                    root = zarr.open_group(zarr_dir, mode='w')
                    root.attrs['multiscales'] = [{
                        'version': '0.4',
                        'axes': [
                            {'name': 'c', 'type': 'channel'},
                            {'name': 'z', 'type': 'space'},
                            {'name': 'y', 'type': 'space'},
                            {'name': 'x', 'type': 'space'}
                        ],
                        'datasets': [{'path': '0'}]
                    }]
                    
                    # Create the dataset
                    dataset = root.create_dataset(
                        '0', 
                        data=img, 
                        chunks=True, 
                        dtype=img.dtype
                    )
                    
                    print(f"Saved OME-Zarr to: {zarr_dir} (direct zarr)")
                    return img
                    
                except Exception as e2:
                    print(f"Direct zarr approach failed: {e2}")
                    raise e2
            
        except ImportError as e:
            print(f"OME-Zarr libraries not available: {e}")
            raise e

    @staticmethod
    def save_as_zarr_simple(image, zarr_path):
        """
        Simple zarr save without OME metadata (from serve2.py)
        """
        try:
            import zarr
            
            # Remove existing zarr directory if it exists
            if os.path.exists(zarr_path):
                import shutil
                shutil.rmtree(zarr_path)
            
            # Save as simple zarr array
            zarr.save_array(zarr_path, image, chunks=True)
            print(f"Saved simple zarr to: {zarr_path}")
            return image
            
        except Exception as e:
            print(f"Simple zarr save failed: {e}")
            raise e

    @staticmethod
    def convert_and_view(input_path, colormap="green", output_zarr=None, simple_zarr=None):
        """
        Convert image and launch napari viewer - main functionality from serve2.py
        """
        try:
            # Set default output paths if not provided
            if output_zarr is None:
                output_zarr = str(Path(input_path).parent / f"{Path(input_path).stem}_output.ome.zarr")
            if simple_zarr is None:
                simple_zarr = str(Path(input_path).parent / f"{Path(input_path).stem}_simple.zarr")
            
            # Load the original image
            original_image = imread(input_path)
            print(f"Original TIFF shape: {original_image.shape}")
            print(f"Original TIFF dtype: {original_image.dtype}")
            print(f"Image min/max: {original_image.min()}/{original_image.max()}")
            
            if original_image is None:
                raise ValueError("Failed to load image")
            
            # Try different conversion methods in order of preference
            processed_image = None
            conversion_method = "Original TIFF"
            
            # Method 1: Try OME-Zarr conversion
            try:
                processed_image = ImageConverter.tiff_to_omezarr_v1(input_path, output_zarr, scale=False)
                conversion_method = "OME-Zarr"
            except Exception as e:
                print(f"OME-Zarr conversion failed: {e}")
                
                # Method 2: Try simple zarr
                try:
                    processed_image = ImageConverter.save_as_zarr_simple(original_image, simple_zarr)
                    conversion_method = "Simple Zarr"
                except Exception as e2:
                    print(f"Simple zarr conversion failed: {e2}")
                    processed_image = original_image
            
            # Launch matplotlib viewer
            try:
                import matplotlib.pyplot as plt  # type: ignore
                import matplotlib.cm as cm  # type: ignore
                
                print(f"Creating matplotlib viewer with {conversion_method}...")
                
                # Create matplotlib figure
                plt.figure(figsize=(10, 8))
                plt.title(f"Fluorescence Viewer - {conversion_method}")
                
                # Display image with appropriate colormap
                if colormap == "green":
                    cmap = "Greens"
                elif colormap == "red":
                    cmap = "Reds"
                elif colormap == "blue":
                    cmap = "Blues"
                else:
                    cmap = colormap
                
                plt.imshow(processed_image, cmap=cmap, vmin=processed_image.min(), vmax=processed_image.max())
                plt.colorbar(label='Intensity')
                plt.xlabel('X')
                plt.ylabel('Y')
                
                # Save the plot to a file
                output_path = str(Path(input_path).parent / f"{Path(input_path).stem}_view.png")
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()  # Close the figure to free memory
                
                return {
                    "status": "success",
                    "message": f"Successfully processed and created viewer for {Path(input_path).name}",
                    "conversion_method": conversion_method,
                    "output_zarr": output_zarr if conversion_method == "OME-Zarr" else None,
                    "simple_zarr": simple_zarr if conversion_method == "Simple Zarr" else None,
                    "image_info": {
                        "shape": original_image.shape,
                        "dtype": str(original_image.dtype),
                        "min_value": int(original_image.min()),
                        "max_value": int(original_image.max())
                    },
                    "viewer_settings": {
                        "colormap": colormap,
                        "contrast_limits": [int(processed_image.min()), int(processed_image.max())],
                        "output_image": output_path
                    }
                }
                
            except ImportError:
                return {
                    "status": "error",
                    "message": "Matplotlib not available. Install with: pip install matplotlib",
                    "conversion_method": conversion_method,
                    "output_zarr": output_zarr if conversion_method == "OME-Zarr" else None,
                    "simple_zarr": simple_zarr if conversion_method == "Simple Zarr" else None
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error in convert_and_view: {str(e)}"
            }

    @staticmethod
    def tiff_to_omezarr(image: np.ndarray, output_path: str, scale: bool = False) -> dict:
        """Convert image array to OME-Zarr format"""
        try:
            # Reshape for OME-Zarr (expects CZYX format)
            if image.ndim == 2:
                reshaped = image[None, None, ...]  # (1, 1, Y, X)
            elif image.ndim == 3:
                reshaped = image[None, ...]  # (1, Z, Y, X)
            else:
                reshaped = image
            
            # Remove existing zarr directory if it exists
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            
            # Try OME-Zarr with proper metadata
            try:
                from ome_zarr.writer import write_image
                from ome_zarr.io import parse_url
                
                # Use zarr.open_group directly instead of parse_url
                group = zarr.open_group(output_path, mode='w')
                
                # Prepare scale factors properly
                scale_factors = [{"x": 2, "y": 2, "z": 2}] if scale else []
                
                write_image(
                    reshaped, 
                    group=group, 
                    axes="czyx", 
                    scale_factors=scale_factors
                )
                method = "OME-Zarr API"
                
            except Exception as e1:
                print(f"OME-Zarr API failed: {e1}")
                # Fallback to direct zarr with OME metadata
                try:
                    root = zarr.open_group(output_path, mode='w')
                    root.attrs['multiscales'] = [{
                        'version': '0.4',
                        'axes': [
                            {'name': 'c', 'type': 'channel'},
                            {'name': 'z', 'type': 'space'},
                            {'name': 'y', 'type': 'space'},
                            {'name': 'x', 'type': 'space'}
                        ],
                        'datasets': [{'path': '0'}]
                    }]
                    
                    root.create_dataset('0', data=reshaped, chunks=True, dtype=reshaped.dtype)
                    method = "Direct Zarr"
                    
                except Exception as e2:
                    print(f"Direct zarr approach failed: {e2}")
                    raise e2
            
            result = {
                "success": True,
                "method": method,
                "output_path": output_path,
                "original_shape": image.shape,
                "zarr_shape": reshaped.shape,
                "chunks": True,
                "dtype": str(image.dtype)
            }
            
            return result
            
        except Exception as e:
            raise ValueError(f"OME-Zarr conversion failed: {str(e)}")
    
    @staticmethod
    def save_simple_zarr(image: np.ndarray, output_path: str) -> dict:
        """Save image as simple zarr array"""
        try:
            if os.path.exists(output_path):
                shutil.rmtree(output_path)
            
            zarr.save_array(output_path, image, chunks=True)
            
            return {
                "success": True,
                "method": "Simple Zarr",
                "output_path": output_path,
                "shape": image.shape,
                "dtype": str(image.dtype)
            }
            
        except Exception as e:
            raise ValueError(f"Simple zarr save failed: {str(e)}")

    @staticmethod
    def launch_matplotlib_viewer(image_path: str, colormap: str = "green") -> dict:
        """Launch matplotlib viewer with an image"""
        try:
            import matplotlib.pyplot as plt  # type: ignore
            import matplotlib.cm as cm  # type: ignore
            
            # Load image
            image, metadata = ImageConverter.load_image(image_path)
            
            # Create matplotlib figure
            plt.figure(figsize=(10, 8))
            plt.title(f"Matplotlib Viewer - {Path(image_path).name}")
            
            # Display image with appropriate colormap
            if colormap == "green":
                cmap = "Greens"
            elif colormap == "red":
                cmap = "Reds"
            elif colormap == "blue":
                cmap = "Blues"
            else:
                cmap = colormap
            
            plt.imshow(image, cmap=cmap, vmin=metadata["min_value"], vmax=metadata["max_value"])
            plt.colorbar(label='Intensity')
            plt.xlabel('X')
            plt.ylabel('Y')
            
            # Save the plot to a file
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_view.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory
            
            return {
                "status": "success",
                "message": f"Created matplotlib viewer for {image_path}",
                "metadata": metadata,
                "viewer_settings": {
                    "colormap": colormap,
                    "contrast_limits": [metadata["min_value"], metadata["max_value"]],
                    "output_image": output_path
                }
            }
            
        except ImportError:
            return {
                "status": "error",
                "message": "Matplotlib not available. Install with: pip install matplotlib"
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to create matplotlib viewer: {str(e)}"
            }

    @staticmethod
    def count_cells(image: np.ndarray, method: str = "watershed", min_distance: int = 10, 
                   threshold_percentile: float = 95.0, min_area: int = 50) -> dict:
        """
        Count cells in a fluorescence image using various methods
        
        Args:
            image: Input image array
            method: Detection method ('watershed', 'peak_detection', 'threshold')
            min_distance: Minimum distance between detected cells
            threshold_percentile: Percentile for thresholding (0-100)
            min_area: Minimum area for cell detection
            
        Returns:
            Dictionary with cell count and detection results
        """
        if not CELL_COUNTING_AVAILABLE:
            return {
                "status": "error",
                "message": "Cell counting libraries not available. Install with: pip install scikit-image scipy"
            }
        
        try:
            # Ensure image is 2D
            if image.ndim > 2:
                if image.ndim == 3:
                    # Take middle slice for 3D images
                    image = image[image.shape[0]//2, :, :]
                else:
                    # Take first channel for multi-channel images
                    image = image[0, :, :] if image.shape[0] == 1 else image[0, 0, :, :]
            
            # Normalize image to 0-1 range
            image_norm = (image - image.min()) / (image.max() - image.min())
            
            if method == "watershed":
                return ImageConverter._watershed_cell_count(image_norm, min_distance, threshold_percentile, min_area)
            elif method == "peak_detection":
                return ImageConverter._peak_detection_cell_count(image_norm, min_distance, threshold_percentile, min_area)
            elif method == "threshold":
                return ImageConverter._threshold_cell_count(image_norm, threshold_percentile, min_area)
            else:
                return {
                    "status": "error",
                    "message": f"Unknown method: {method}. Use 'watershed', 'peak_detection', or 'threshold'"
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Cell counting failed: {str(e)}"
            }

    @staticmethod
    def _watershed_cell_count(image: np.ndarray, min_distance: int, threshold_percentile: float, min_area: int) -> dict:
        """Watershed-based cell counting"""
        try:
            # Apply Gaussian blur to reduce noise
            from skimage.filters import gaussian
            image_blur = gaussian(image, sigma=1)
            
            # Create binary mask using threshold
            threshold = np.percentile(image_blur, threshold_percentile)
            binary = image_blur > threshold
            
            # Remove small objects
            binary = morphology.remove_small_objects(binary, min_size=min_area)
            
            # Distance transform
            distance = ndimage.distance_transform_edt(binary)
            
            # Find local maxima
            coords = peak_local_maxima(distance, min_distance=min_distance, labels=binary)
            
            # Create markers for watershed
            markers = np.zeros_like(binary, dtype=bool)
            if len(coords) > 0:
                markers[coords[:, 0], coords[:, 1]] = True
            markers = measure.label(markers)
            
            # Watershed segmentation
            distance_array = np.array(distance, dtype=float)
            labels = segmentation.watershed(-distance_array, markers, mask=binary)
            
            # Count cells
            cell_count = len(np.unique(labels)) - 1  # Subtract 1 for background
            
            return {
                "status": "success",
                "method": "watershed",
                "cell_count": cell_count,
                "detection_coordinates": coords.tolist(),
                "parameters": {
                    "min_distance": min_distance,
                    "threshold_percentile": threshold_percentile,
                    "min_area": min_area
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Watershed method failed: {str(e)}"
            }

    @staticmethod
    def _peak_detection_cell_count(image: np.ndarray, min_distance: int, threshold_percentile: float, min_area: int) -> dict:
        """Peak detection-based cell counting"""
        try:
            # Apply Gaussian blur
            from skimage.filters import gaussian
            image_blur = gaussian(image, sigma=1)
            
            # Find local maxima
            coords = peak_local_maxima(image_blur, min_distance=min_distance, 
                                     threshold_abs=np.percentile(image_blur, threshold_percentile))
            
            # Filter by area (create regions around peaks)
            binary = image_blur > np.percentile(image_blur, threshold_percentile)
            labels = measure.label(binary)
            labels = np.asarray(labels)  # Ensure labels is a numpy array for type checker
            
            # Count regions that contain peaks
            valid_regions = set()
            for coord in coords:
                if len(coord) >= 2:
                    x, y = int(coord[0]), int(coord[1])
                    region_label = labels[x, y]
                    if region_label > 0:
                        region_area = np.sum(labels == region_label)
                        if region_area >= min_area:
                            valid_regions.add(region_label)
            
            cell_count = len(valid_regions)
            
            return {
                "status": "success",
                "method": "peak_detection",
                "cell_count": cell_count,
                "detection_coordinates": coords.tolist(),
                "parameters": {
                    "min_distance": min_distance,
                    "threshold_percentile": threshold_percentile,
                    "min_area": min_area
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Peak detection method failed: {str(e)}"
            }

    @staticmethod
    def _threshold_cell_count(image: np.ndarray, threshold_percentile: float, min_area: int) -> dict:
        """Simple threshold-based cell counting"""
        try:
            # Apply threshold
            threshold = np.percentile(image, threshold_percentile)
            binary = image > threshold
            
            # Remove small objects
            binary = morphology.remove_small_objects(binary, min_size=min_area)
            
            # Label connected components
            labels = measure.label(binary)
            labels = np.asarray(labels)  # Ensure labels is a numpy array
            cell_count = len(np.unique(labels)) - 1  # Subtract 1 for background
            
            # Find centroids
            regions = measure.regionprops(labels)
            centroids = [region.centroid for region in regions]
            
            return {
                "status": "success",
                "method": "threshold",
                "cell_count": cell_count,
                "detection_coordinates": centroids,
                "parameters": {
                    "threshold_percentile": threshold_percentile,
                    "min_area": min_area
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Threshold method failed: {str(e)}"
            }

    @staticmethod
    def convert_and_count_cells(input_path: str, method: str = "watershed", min_distance: int = 10,
                              threshold_percentile: float = 95.0, min_area: int = 50,
                              colormap: str = "green", output_zarr: Optional[str] = None) -> dict:
        """
        Convert image to OME-Zarr and count cells in one operation
        """
        try:
            # Load image
            image, metadata = ImageConverter.load_image(input_path)
            
            # Convert to OME-Zarr if output path provided
            conversion_result = None
            if output_zarr:
                conversion_result = ImageConverter.tiff_to_omezarr(image, output_zarr, scale=False)
            
            # Count cells
            cell_result = ImageConverter.count_cells(image, method, min_distance, threshold_percentile, min_area)
            
            # Launch matplotlib viewer
            viewer_result = ImageConverter.launch_matplotlib_viewer(input_path, colormap)
            
            return {
                "status": "success",
                "message": f"Processed {Path(input_path).name} with cell counting",
                "conversion": conversion_result,
                "cell_counting": cell_result,
                "viewer": viewer_result,
                "metadata": metadata
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Convert and count failed: {str(e)}"
            }


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="convert_and_view",
            description="Convert a fluorescence image to OME-Zarr format and create matplotlib viewer",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input fluorescence image file"
                    },
                    "colormap": {
                        "type": "string",
                        "description": "Colormap for fluorescence (e.g., 'green', 'red', 'blue')",
                        "default": "green"
                    },
                    "output_zarr": {
                        "type": "string",
                        "description": "Path for output OME-Zarr directory (optional)"
                    },
                    "simple_zarr": {
                        "type": "string",
                        "description": "Path for output simple zarr directory (optional)"
                    }
                },
                "required": ["input_path"]
            }
        ),
        types.Tool(
            name="load_image",
            description="Load an image file and get metadata",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="convert_to_omezarr",
            description="Convert a TIFF image to OME-Zarr format",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input TIFF file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for output OME-Zarr directory"
                    },
                    "scale": {
                        "type": "boolean",
                        "description": "Whether to create multi-scale pyramid",
                        "default": False
                    }
                },
                "required": ["input_path", "output_path"]
            }
        ),
        types.Tool(
            name="convert_to_simple_zarr",
            description="Convert an image to simple zarr format",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input image file"
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Path for output zarr directory"
                    }
                },
                "required": ["input_path", "output_path"]
            }
        ),
        types.Tool(
            name="launch_matplotlib",
            description="Create matplotlib viewer with an image",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "Path to image file to view"
                    },
                    "colormap": {
                        "type": "string",
                        "description": "Colormap to use (e.g., 'green', 'viridis', 'gray')",
                        "default": "green"
                    }
                },
                "required": ["image_path"]
            }
        ),
        types.Tool(
            name="get_image_info",
            description="Get detailed information about an image file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the image file"
                    }
                },
                "required": ["file_path"]
            }
        ),
        types.Tool(
            name="process_fluorescence_image",
            description="Process a fluorescence image with optimal settings for visualization",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input fluorescence image file"
                    },
                    "output_zarr_path": {
                        "type": "string",
                        "description": "Path for output OME-Zarr directory"
                    },
                    "colormap": {
                        "type": "string",
                        "description": "Colormap for fluorescence (e.g., 'green', 'red', 'blue')",
                        "default": "green"
                    }
                },
                "required": ["input_path", "output_zarr_path"]
            }
        ),
        types.Tool(
            name="count_cells",
            description="Count cells in a fluorescence image using various detection methods",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input fluorescence image file"
                    },
                    "method": {
                        "type": "string",
                        "description": "Cell detection method: 'watershed', 'peak_detection', or 'threshold'",
                        "default": "watershed"
                    },
                    "min_distance": {
                        "type": "integer",
                        "description": "Minimum distance between detected cells (pixels)",
                        "default": 10
                    },
                    "threshold_percentile": {
                        "type": "number",
                        "description": "Percentile for thresholding (0-100)",
                        "default": 95.0
                    },
                    "min_area": {
                        "type": "integer",
                        "description": "Minimum area for cell detection (pixels)",
                        "default": 50
                    }
                },
                "required": ["input_path"]
            }
        ),
        types.Tool(
            name="convert_and_count_cells",
            description="Convert image to OME-Zarr format and count cells in one operation",
            inputSchema={
                "type": "object",
                "properties": {
                    "input_path": {
                        "type": "string",
                        "description": "Path to input fluorescence image file"
                    },
                    "output_zarr": {
                        "type": "string",
                        "description": "Path for output OME-Zarr directory (optional)"
                    },
                    "method": {
                        "type": "string",
                        "description": "Cell detection method: 'watershed', 'peak_detection', or 'threshold'",
                        "default": "watershed"
                    },
                    "min_distance": {
                        "type": "integer",
                        "description": "Minimum distance between detected cells (pixels)",
                        "default": 10
                    },
                    "threshold_percentile": {
                        "type": "number",
                        "description": "Percentile for thresholding (0-100)",
                        "default": 95.0
                    },
                    "min_area": {
                        "type": "integer",
                        "description": "Minimum area for cell detection (pixels)",
                        "default": 50
                    },
                    "colormap": {
                        "type": "string",
                        "description": "Colormap for fluorescence (e.g., 'green', 'red', 'blue')",
                        "default": "green"
                    }
                },
                "required": ["input_path"]
            }
        )
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "convert_and_view":
            input_path = arguments["input_path"]
            colormap = arguments.get("colormap", "green")
            output_zarr = arguments.get("output_zarr")
            simple_zarr = arguments.get("simple_zarr")
            
            result = ImageConverter.convert_and_view(
                input_path=input_path,
                colormap=colormap,
                output_zarr=output_zarr,
                simple_zarr=simple_zarr
            )
            
        elif name == "load_image":
            file_path = arguments["file_path"]
            image, metadata = ImageConverter.load_image(file_path)
            
            result = {
                "status": "success",
                "message": f"Successfully loaded image from {file_path}",
                "metadata": metadata
            }
            
        elif name == "convert_to_omezarr":
            input_path = arguments["input_path"]
            output_path = arguments["output_path"]
            scale = arguments.get("scale", False)
            
            # Load image
            image, _ = ImageConverter.load_image(input_path)
            
            # Convert to OME-Zarr
            result = ImageConverter.tiff_to_omezarr(image, output_path, scale)
            result["message"] = f"Successfully converted {input_path} to OME-Zarr format"
            
        elif name == "convert_to_simple_zarr":
            input_path = arguments["input_path"]
            output_path = arguments["output_path"]
            
            # Load image
            image, _ = ImageConverter.load_image(input_path)
            
            # Convert to simple zarr
            result = ImageConverter.save_simple_zarr(image, output_path)
            result["message"] = f"Successfully converted {input_path} to simple zarr format"
            
        elif name == "launch_matplotlib":
            image_path = arguments["image_path"]
            colormap = arguments.get("colormap", "green")
            
            result = ImageConverter.launch_matplotlib_viewer(image_path, colormap)
                
        elif name == "get_image_info":
            file_path = arguments["file_path"]
            image, metadata = ImageConverter.load_image(file_path)
            
            # Add more detailed info
            metadata.update({
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "dimensions": len(image.shape),
                "total_pixels": int(np.prod(image.shape)),
                "memory_usage_mb": round(image.nbytes / (1024*1024), 2)
            })
            
            result = {
                "status": "success",
                "message": f"Image information for {Path(file_path).name}",
                "info": metadata
            }
            
        elif name == "process_fluorescence_image":
            input_path = arguments["input_path"]
            output_zarr_path = arguments["output_zarr_path"]
            colormap = arguments.get("colormap", "green")
            
            # Load and process the fluorescence image
            image, metadata = ImageConverter.load_image(input_path)
            
            # Convert to OME-Zarr
            conversion_result = ImageConverter.tiff_to_omezarr(image, output_zarr_path, scale=False)
            
            # Launch matplotlib viewer
            viewer_result = ImageConverter.launch_matplotlib_viewer(input_path, colormap)
            
            result = {
                "status": "success",
                "message": f"Processed fluorescence image {Path(input_path).name}",
                "conversion": conversion_result,
                "viewer": viewer_result,
                "metadata": metadata
            }
            
        elif name == "count_cells":
            input_path = arguments["input_path"]
            method = arguments.get("method", "watershed")
            min_distance = arguments.get("min_distance", 10)
            threshold_percentile = arguments.get("threshold_percentile", 95.0)
            min_area = arguments.get("min_area", 50)
            
            # Load image first
            image, metadata = ImageConverter.load_image(input_path)
            result = ImageConverter.count_cells(image, method, min_distance, threshold_percentile, min_area)
            
        elif name == "convert_and_count_cells":
            input_path = arguments["input_path"]
            output_zarr = arguments.get("output_zarr")
            method = arguments.get("method", "watershed")
            min_distance = arguments.get("min_distance", 10)
            threshold_percentile = arguments.get("threshold_percentile", 95.0)
            min_area = arguments.get("min_area", 50)
            colormap = arguments.get("colormap", "green")
            
            result = ImageConverter.convert_and_count_cells(input_path, method, min_distance, threshold_percentile, min_area, colormap, output_zarr or "")
            
        else:
            result = {
                "status": "error",
                "message": f"Unknown tool: {name}"
            }
            
    except Exception as e:
        result = {
            "status": "error",
            "message": f"Error executing {name}: {str(e)}"
        }
    
    return [types.TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Main server entry point"""
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="image-converter",
                server_version="1.0.0",
                capabilities=ServerCapabilities()
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
