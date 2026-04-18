#!/usr/bin/env python3
"""
Start server programmatically to debug issues
"""

if __name__ == "__main__":
    try:
        import uvicorn
        from app.main import app
        
        print("Starting server...")
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
        
    except Exception as e:
        print(f"Error starting server: {e}")
        import traceback
        traceback.print_exc()