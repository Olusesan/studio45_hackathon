## Prerequisites

1. **Install Clause Desktop**
   - Download from: https://clause.ai/
   - Install and launch Clause Desktop

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

### Option 1: Use the provided config file

1. Copy `clause_config.json` to your Clause Desktop configuration directory:
   - **macOS**: `~/Library/Application Support/Clause/`
   - **Windows**: `%APPDATA%\Clause\`
   - **Linux**: `~/.config/Clause/`

2. Restart Clause Desktop

### Option 2: Configure through Clause Desktop UI

1. Open Clause Desktop
2. Go to Settings → MCP Servers
3. Add a new server with these settings:
   - **Name**: `image-converter`
   - **Command**: `python`
   - **Arguments**: `["server.py"]`
   - **Working Directory**: Path to your project folder
   - **Environment Variables**: 
     - `PYTHONPATH`: `.`

## Testing the Integration

### 1. Verify Server Connection

In Clause Desktop, you should see the `image-converter` server listed in the MCP servers section. The server should show as "Connected" if everything is working correctly.

### 2. Test Basic Functionality

Try these commands in Clause Desktop:

#### Get Image Information
```
Use the get_image_info tool to analyze the fluorescence image at /Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff
```

#### Convert to OME-Zarr
```
Convert the fluorescence image at /Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff to OME-Zarr format and save it to /Users/sesan/Desktop/output_image.ome.zarr
```

#### Launch Napari Viewer
```
Launch napari viewer to display the fluorescence image at /Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff with a green colormap
```

#### Process Fluorescence Image
```
Process the fluorescence image at /Users/sesan/Desktop/A1_0_0_Fluorescence_405_nm_Ex.tiff with optimal settings, convert it to OME-Zarr format, and launch napari viewer
```

### 3. Available Tools

The MCP server provides these tools:

1. **`load_image`** - Load an image and get metadata
2. **`convert_to_omezarr`** - Convert TIFF to OME-Zarr format
3. **`convert_to_simple_zarr`** - Convert to simple zarr format
4. **`launch_napari`** - Launch napari viewer
5. **`get_image_info`** - Get detailed image information
6. **`process_fluorescence_image`** - All-in-one fluorescence processing

## Troubleshooting

### Server Not Connecting

1. **Check Python path**: Ensure `python` is in your PATH
2. **Check dependencies**: Run `python test_mcp_server.py` to verify all dependencies are installed
3. **Check file permissions**: Ensure `server.py` is executable
4. **Check working directory**: Make sure Clause Desktop is running from the correct directory

### Import Errors

If you see import errors, try:
```bash
pip install -r requirements.txt
pip install 'napari[all]'
```

### Napari Issues

If napari doesn't launch:
1. Install Qt bindings: `pip install pyqt5`
2. Check if you have a display server running (for headless systems)

### Debug Mode

To run the server in debug mode, modify the config:
```json
{
  "mcpServers": {
    "image-converter": {
      "command": "python",
      "args": ["-u", "server.py"],
      "env": {
        "PYTHONPATH": ".",
        "PYTHONUNBUFFERED": "1"
      }
    }
  }
}
```

## Example Conversations

### Basic Image Analysis
**User**: "Analyze the fluorescence image on my desktop"

**Assistant**: I'll help you analyze the fluorescence image. Let me get detailed information about it.

*[Uses `get_image_info` tool]*

The image has the following properties:
- Shape: (2048, 2048)
- Data type: uint16
- Min/Max values: 0/65535
- File size: 8.4 MB
- Total pixels: 4,194,304

### Image Conversion
**User**: "Convert this image to OME-Zarr format for scientific data sharing"

**Assistant**: I'll convert the image to OME-Zarr format, which is ideal for scientific data sharing and visualization.

*[Uses `convert_to_omezarr` tool]*

Successfully converted the image to OME-Zarr format:
- Output: `/Users/sesan/Desktop/output_image.ome.zarr`
- Method: OME-Zarr API
- Original shape: (2048, 2048)
- Zarr shape: (1, 1, 2048, 2048)

### Visualization
**User**: "Show me the image in napari with a green colormap"

**Assistant**: I'll launch napari viewer to display the fluorescence image with a green colormap, which is optimal for fluorescence microscopy data.

*[Uses `launch_napari` tool]*

Napari viewer has been launched with:
- Colormap: green
- Contrast limits: [0, 65535]
- The viewer window should now be open for interactive exploration

## Advanced Usage

### Batch Processing
You can ask Clause to process multiple images or perform complex workflows:

**User**: "Process all TIFF files in my Desktop folder and convert them to OME-Zarr format"

### Custom Analysis
**User**: "Analyze the fluorescence intensity distribution in this image"

### Data Export
**User**: "Convert this image to both OME-Zarr and simple zarr formats for different use cases" 