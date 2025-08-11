#!/usr/bin/env python3
"""
API Startup Script für BU-Processor
===================================

Einfaches Script zum Starten der FastAPI-REST-API.
Kann direkt ausgeführt werden oder als Service deployed werden.
"""

import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Start BU-Processor REST API")
    parser.add_argument("--host", default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"])
    
    args = parser.parse_args()
    
    try:
        import uvicorn
        
        print(f"🚀 Starting BU-Processor API Server")
        print(f"📡 Host: {args.host}")
        print(f"🔌 Port: {args.port}")
        print(f"🔄 Auto-reload: {'✅' if args.reload else '❌'}")
        print(f"👥 Workers: {args.workers}")
        print(f"📚 API Documentation: http://{args.host}:{args.port}/docs")
        print(f"🩺 Health Check: http://{args.host}:{args.port}/health")
        print(f"📄 API Info: http://{args.host}:{args.port}/")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 60)
        
        uvicorn.run(
            "bu_processor.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level,
            access_log=True
        )
        
    except ImportError:
        print("❌ uvicorn not installed. Install with:")
        print("pip install uvicorn[standard]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
