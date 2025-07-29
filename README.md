# Image Converter MCP Server

An MCP (Model Context Protocol) server for image processing and visualization, specifically designed for scientific images like fluorescence microscopy data.

## Features

- **Image Loading**: Load various image formats (TIFF, PNG, JPEG, etc.)
- **OME-Zarr Conversion**: Convert images to OME-Zarr format for scientific data sharing
- **Simple Zarr**: Save images in simple zarr format
- **Matplotlib Integration**: Create static image visualizations with matplotlib
- **Fluorescence Image Processing**: Specialized tools for fluorescence microscopy data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For matplotlib visualization (included in requirements.txt):
```bash
pip install matplotlib
```

## Usage

### Running the MCP Server

```bash
python server.py
```

The server communicates via stdin/stdout and can be integrated with MCP clients.

### Available Tools

#### 1. `load_image`
Load an image file and get metadata.

**Parameters:**
- `file_path` (string): Path to the image file

**Example:**
```json
{
  "file_path": "/path/to/image.tiff"
}
```

#### 2. `convert_to_omezarr`
Convert a TIFF image to OME-Zarr format.

**Parameters:**
- `input_path` (string): Path to input TIFF file
- `output_path` (string): Path for output OME-Zarr directory
- `scale` (boolean, optional): Whether to create multi-scale pyramid (default: false)

**Example:**
```json
{
  "input_path": "/path/to/input.tiff",
  "output_path": "/path/to/output.ome.zarr",
  "scale": true
}
```

#### 3. `convert_to_simple_zarr`
Convert an image to simple zarr format.

**Parameters:**
- `input_path` (string): Path to input image file
- `output_path` (string): Path for output zarr directory

#### 4. `launch_matplotlib`
Create matplotlib viewer with an image.

**Parameters:**
- `image_path` (string): Path to image file to view
- `colormap` (string, optional): Colormap to use (default: "green")

**Example:**
```json
{
  "image_path": "/path/to/image.tiff",
  "colormap": "green"
}
```

#### 5. `get_image_info`
Get detailed information about an image file.

**Parameters:**
- `file_path` (string): Path to the image file

#### 6. `process_fluorescence_image`
Process a fluorescence image with optimal settings for visualization.

**Parameters:**
- `input_path` (string): Path to input fluorescence image file
- `output_zarr_path` (string): Path for output OME-Zarr directory
- `colormap` (string, optional): Colormap for fluorescence (default: "green")

## Example Workflow

1. **Load and analyze a fluorescence image:**
```json
{
  "tool": "get_image_info",
  "arguments": {
    "file_path": "/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff"
  }
}
```

2. **Convert to OME-Zarr format:**
```json
{
  "tool": "convert_to_omezarr",
  "arguments": {
    "input_path": "/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff",
    "output_path": "/Users/sesan/Desktop/output_image.ome.zarr"
  }
}
```

3. **Create matplotlib viewer:**
```json
{
  "tool": "launch_matplotlib",
  "arguments": {
    "image_path": "/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff",
    "colormap": "green"
  }
}
```

4. **Process fluorescence image (all-in-one):**
```json
{
  "tool": "process_fluorescence_image",
  "arguments": {
    "input_path": "/Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff",
    "output_zarr_path": "/Users/sesan/Desktop/output_image.ome.zarr",
    "colormap": "green"
  }
}
```

## Integration with MCP Clients

This server can be integrated with any MCP-compatible client. The server provides tools for:

- Image format conversion
- Scientific data visualization
- Metadata extraction
- Static image visualization

## Error Handling

The server includes robust error handling for:
- Missing files
- Unsupported image formats
- OME-Zarr conversion failures
- Matplotlib import issues
- Memory constraints

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black server.py
```

### Linting
```bash
flake8 server.py
```

## Troubleshooting

### Matplotlib Issues
If matplotlib doesn't work, ensure it's installed:
```bash
pip install matplotlib
```

### OME-Zarr Issues
If OME-Zarr conversion fails, the server will fall back to direct zarr creation with OME metadata.

### Memory Issues
For large images, consider using the `scale` parameter to create multi-scale pyramids for better performance.

## License

This project is open source and available under the MIT License. 