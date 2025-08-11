#!/usr/bin/env python3
"""
🌐 CHATBOT WEB INTERFACE - FastAPI Implementation
===============================================

Web-basierte Schnittstelle für den BU-Processor Chatbot mit:
- REST API für Chatbot-Interaktionen
- WebSocket für Real-time Chat
- Web UI für Interactive Chat
- Monitoring Dashboard
- Multi-Session Management
"""

import asyncio
import json
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# FastAPI und Web-Dependencies
try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("⚠️  FastAPI nicht installiert. Installiere mit: pip install fastapi uvicorn")

# Pydantic für Request/Response Models
try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    print("⚠️  Pydantic nicht installiert. Installiere mit: pip install pydantic")

# Chatbot Integration
try:
    from .chatbot_integration import BUProcessorChatbot, ChatbotConfig, ConversationManager
    CHATBOT_AVAILABLE = True
except ImportError:
    CHATBOT_AVAILABLE = False
    print("⚠️  Chatbot Module nicht gefunden")

import structlog

# Logging Setup
logger = structlog.get_logger("chatbot_web_interface")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ChatRequest(BaseModel):
    """Chat Request Model"""
    message: str = Field(..., description="User message", min_length=1, max_length=2000)
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    include_context: bool = Field(True, description="Whether to include RAG context")
    model: Optional[str] = Field("gpt-4o-mini", description="OpenAI model to use")

class ChatResponse(BaseModel):
    """Chat Response Model"""
    response: str = Field(..., description="Chatbot response")
    session_id: str = Field(..., description="Session ID")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    tokens_used: int = Field(..., description="Number of tokens used")
    context_used: bool = Field(..., description="Whether RAG context was used")
    sources: List[str] = Field(default_factory=list, description="Source documents")
    conversation_turn: int = Field(..., description="Turn number in conversation")
    timestamp: str = Field(..., description="Response timestamp")

class SessionInfo(BaseModel):
    """Session Information Model"""
    session_id: str
    created_at: str
    last_activity: str
    total_turns: int
    total_tokens: int
    active: bool

class SystemStatus(BaseModel):
    """System Status Model"""
    status: str
    chatbot_available: bool
    rag_enabled: bool
    active_sessions: int
    total_queries_today: int
    avg_response_time_ms: float
    uptime_seconds: float

# =============================================================================
# SESSION MANAGER
# =============================================================================

class ChatSessionManager:
    """Manages multiple chat sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.daily_stats = {
            "total_queries": 0,
            "total_tokens": 0,
            "total_response_time": 0.0,
            "start_time": time.time()
        }
        
    def create_session(self, config: Optional[ChatbotConfig] = None) -> str:
        """Creates a new chat session"""
        session_id = str(uuid.uuid4())
        
        try:
            chatbot = BUProcessorChatbot(config or ChatbotConfig())
            
            self.sessions[session_id] = {
                "chatbot": chatbot,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "active": True,
                "websocket": None
            }
            
            logger.info("New chat session created", session_id=session_id)
            return session_id
            
        except Exception as e:
            logger.error("Failed to create chat session", error=str(e))
            raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")
    
    def get_session(self, session_id: str) -> Optional[BUProcessorChatbot]:
        """Gets chatbot for session"""
        session = self.sessions.get(session_id)
        if session and session["active"]:
            session["last_activity"] = datetime.now()
            return session["chatbot"]
        return None
    
    def close_session(self, session_id: str):
        """Closes a chat session"""
        if session_id in self.sessions:
            self.sessions[session_id]["active"] = False
            logger.info("Chat session closed", session_id=session_id)
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Cleanup old sessions"""
        now = datetime.now()
        inactive_sessions = []
        
        for session_id, session_data in self.sessions.items():
            age = (now - session_data["last_activity"]).total_seconds() / 3600
            if age > max_age_hours:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            self.close_session(session_id)
        
        if inactive_sessions:
            logger.info("Cleaned up inactive sessions", count=len(inactive_sessions))
    
    def get_session_info(self, session_id: str) -> Optional[SessionInfo]:
        """Gets session information"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        chatbot = session["chatbot"]
        stats = chatbot.get_stats()
        
        return SessionInfo(
            session_id=session_id,
            created_at=session["created_at"].isoformat(),
            last_activity=session["last_activity"].isoformat(),
            total_turns=stats["conversation_stats"]["total_turns"],
            total_tokens=stats["chatbot_stats"]["total_tokens_used"],
            active=session["active"]
        )
    
    def get_all_sessions(self) -> List[SessionInfo]:
        """Gets information for all sessions"""
        return [
            self.get_session_info(session_id) 
            for session_id in self.sessions.keys()
            if self.get_session_info(session_id)
        ]
    
    def update_daily_stats(self, tokens_used: int, response_time_ms: float):
        """Updates daily statistics"""
        self.daily_stats["total_queries"] += 1
        self.daily_stats["total_tokens"] += tokens_used
        self.daily_stats["total_response_time"] += response_time_ms

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

if FASTAPI_AVAILABLE and PYDANTIC_AVAILABLE:
    # Initialize FastAPI
    app = FastAPI(
        title="BU-Processor Chatbot API",
        description="Web API für den BU-Processor Chatbot mit RAG Integration",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Session Manager
    session_manager = ChatSessionManager()
    
    # Startup Event
    @app.on_event("startup")
    async def startup_event():
        logger.info("Chatbot Web API starting up")
        
        # Cleanup task
        async def cleanup_task():
            while True:
                await asyncio.sleep(3600)  # Every hour
                session_manager.cleanup_inactive_sessions()
        
        asyncio.create_task(cleanup_task())
    
    # =============================================================================
    # REST API ENDPOINTS
    # =============================================================================
    
    @app.get("/", response_class=HTMLResponse)
    async def web_interface():
        """Serves the web chat interface"""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>BU-Processor Chatbot</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; min-height: 100vh; display: flex; flex-direction: column; }
                .header { background: #2563eb; color: white; padding: 1rem; text-align: center; }
                .header h1 { margin: 0; font-size: 1.5rem; }
                .chat-container { flex: 1; padding: 1rem; overflow-y: auto; }
                .message { margin: 1rem 0; padding: 0.75rem 1rem; border-radius: 1rem; max-width: 70%; }
                .user-message { background: #2563eb; color: white; margin-left: auto; }
                .bot-message { background: #e5e7eb; color: #374151; }
                .input-container { padding: 1rem; border-top: 1px solid #e5e7eb; }
                .input-row { display: flex; gap: 0.5rem; }
                #messageInput { flex: 1; padding: 0.75rem; border: 1px solid #d1d5db; border-radius: 0.5rem; }
                #sendButton { padding: 0.75rem 1.5rem; background: #2563eb; color: white; border: none; border-radius: 0.5rem; cursor: pointer; }
                #sendButton:hover { background: #1d4ed8; }
                .typing { opacity: 0.7; font-style: italic; }
                .sources { font-size: 0.8rem; color: #6b7280; margin-top: 0.5rem; }
                .stats { font-size: 0.8rem; color: #6b7280; margin-top: 0.25rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 BU-Processor Chatbot</h1>
                    <p>Stelle Fragen zu Berufsunfähigkeitsversicherungen</p>
                </div>
                
                <div class="chat-container" id="chatContainer">
                    <div class="message bot-message">
                        <div>Hallo! Ich bin dein BU-Versicherungs-Assistent. Wie kann ich dir helfen?</div>
                    </div>
                </div>
                
                <div class="input-container">
                    <div class="input-row">
                        <input type="text" id="messageInput" placeholder="Stelle deine Frage..." onkeypress="handleKeyPress(event)">
                        <button id="sendButton" onclick="sendMessage()">Senden</button>
                    </div>
                </div>
            </div>
            
            <script>
                let sessionId = null;
                
                async function sendMessage() {
                    const input = document.getElementById('messageInput');
                    const message = input.value.trim();
                    
                    if (!message) return;
                    
                    // Add user message to chat
                    addMessage(message, 'user');
                    input.value = '';
                    
                    // Show typing indicator
                    const typingDiv = addMessage('🤔 Denke nach...', 'bot', true);
                    
                    try {
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                message: message,
                                session_id: sessionId,
                                include_context: true
                            })
                        });
                        
                        const data = await response.json();
                        
                        // Remove typing indicator
                        typingDiv.remove();
                        
                        if (response.ok) {
                            sessionId = data.session_id;
                            
                            // Add bot response
                            const botDiv = addMessage(data.response, 'bot');
                            
                            // Add sources if available
                            if (data.sources && data.sources.length > 0) {
                                const sourcesDiv = document.createElement('div');
                                sourcesDiv.className = 'sources';
                                sourcesDiv.textContent = '📚 Quellen: ' + data.sources.join(', ');
                                botDiv.appendChild(sourcesDiv);
                            }
                            
                            // Add stats
                            const statsDiv = document.createElement('div');
                            statsDiv.className = 'stats';
                            statsDiv.textContent = `⚡ ${data.response_time_ms.toFixed(0)}ms | 🎫 ${data.tokens_used} tokens`;
                            botDiv.appendChild(statsDiv);
                            
                        } else {
                            addMessage('❌ Fehler: ' + (data.detail || 'Unbekannter Fehler'), 'bot');
                        }
                        
                    } catch (error) {
                        typingDiv.remove();
                        addMessage('❌ Verbindungsfehler: ' + error.message, 'bot');
                    }
                }
                
                function addMessage(text, sender, isTemporary = false) {
                    const chatContainer = document.getElementById('chatContainer');
                    const messageDiv = document.createElement('div');
                    messageDiv.className = `message ${sender}-message`;
                    
                    if (isTemporary) {
                        messageDiv.classList.add('typing');
                    }
                    
                    const textDiv = document.createElement('div');
                    textDiv.textContent = text;
                    messageDiv.appendChild(textDiv);
                    
                    chatContainer.appendChild(messageDiv);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                    
                    return messageDiv;
                }
                
                function handleKeyPress(event) {
                    if (event.key === 'Enter') {
                        sendMessage();
                    }
                }
                
                // Focus input on load
                document.getElementById('messageInput').focus();
            </script>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content)
    
    @app.post("/api/chat", response_model=ChatResponse)
    async def chat_endpoint(request: ChatRequest):
        """Main chat endpoint"""
        try:
            # Get or create session
            session_id = request.session_id
            if not session_id or not session_manager.get_session(session_id):
                session_id = session_manager.create_session()
            
            chatbot = session_manager.get_session(session_id)
            if not chatbot:
                raise HTTPException(status_code=404, detail="Session not found")
            
            # Update model if specified
            if request.model and request.model != chatbot.config.model:
                chatbot.config.model = request.model
            
            # Process chat message
            start_time = time.time()
            result = await chatbot.chat(request.message, request.include_context)
            processing_time = (time.time() - start_time) * 1000
            
            # Update daily stats
            session_manager.update_daily_stats(
                result.get("tokens_used", 0),
                processing_time
            )
            
            # Get conversation stats
            stats = chatbot.get_stats()
            conversation_turn = stats["conversation_stats"]["total_turns"]
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            return ChatResponse(
                response=result["response"],
                session_id=session_id,
                response_time_ms=result["response_time_ms"],
                tokens_used=result["tokens_used"],
                context_used=result["context_used"],
                sources=result.get("sources", []),
                conversation_turn=conversation_turn,
                timestamp=datetime.now().isoformat()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Chat endpoint error", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/status", response_model=SystemStatus)
    async def system_status():
        """System status endpoint"""
        uptime = time.time() - session_manager.daily_stats["start_time"]
        avg_response_time = 0.0
        
        if session_manager.daily_stats["total_queries"] > 0:
            avg_response_time = (
                session_manager.daily_stats["total_response_time"] / 
                session_manager.daily_stats["total_queries"]
            )
        
        return SystemStatus(
            status="healthy",
            chatbot_available=CHATBOT_AVAILABLE,
            rag_enabled=True,  # Assuming RAG is enabled
            active_sessions=len([s for s in session_manager.sessions.values() if s["active"]]),
            total_queries_today=session_manager.daily_stats["total_queries"],
            avg_response_time_ms=avg_response_time,
            uptime_seconds=uptime
        )
    
    @app.get("/api/sessions", response_model=List[SessionInfo])
    async def list_sessions():
        """List all chat sessions"""
        return session_manager.get_all_sessions()
    
    @app.get("/api/sessions/{session_id}", response_model=SessionInfo)
    async def get_session_info(session_id: str):
        """Get information about a specific session"""
        info = session_manager.get_session_info(session_id)
        if not info:
            raise HTTPException(status_code=404, detail="Session not found")
        return info
    
    @app.delete("/api/sessions/{session_id}")
    async def close_session(session_id: str):
        """Close a chat session"""
        if session_id not in session_manager.sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_manager.close_session(session_id)
        return {"message": "Session closed successfully"}
    
    @app.post("/api/sessions/{session_id}/reset")
    async def reset_session(session_id: str):
        """Reset conversation history for a session"""
        chatbot = session_manager.get_session(session_id)
        if not chatbot:
            raise HTTPException(status_code=404, detail="Session not found")
        
        chatbot.reset_conversation()
        return {"message": "Session reset successfully"}
    
    # =============================================================================
    # WEBSOCKET ENDPOINT
    # =============================================================================
    
    @app.websocket("/ws/{session_id}")
    async def websocket_endpoint(websocket: WebSocket, session_id: str):
        """WebSocket endpoint for real-time chat"""
        await websocket.accept()
        
        try:
            # Get or create session
            chatbot = session_manager.get_session(session_id)
            if not chatbot:
                session_id = session_manager.create_session()
                chatbot = session_manager.get_session(session_id)
            
            # Store websocket in session
            session_manager.sessions[session_id]["websocket"] = websocket
            
            await websocket.send_json({
                "type": "session_info",
                "session_id": session_id,
                "message": "WebSocket connection established"
            })
            
            while True:
                # Receive message
                data = await websocket.receive_json()
                message = data.get("message", "")
                
                if not message:
                    continue
                
                # Send typing indicator
                await websocket.send_json({
                    "type": "typing",
                    "message": "🤔 Denke nach..."
                })
                
                # Process message
                try:
                    result = await chatbot.chat(message, data.get("include_context", True))
                    
                    if "error" in result:
                        await websocket.send_json({
                            "type": "error",
                            "message": result["error"]
                        })
                    else:
                        await websocket.send_json({
                            "type": "response",
                            "message": result["response"],
                            "response_time_ms": result["response_time_ms"],
                            "tokens_used": result["tokens_used"],
                            "context_used": result["context_used"],
                            "sources": result.get("sources", [])
                        })
                        
                        # Update stats
                        session_manager.update_daily_stats(
                            result.get("tokens_used", 0),
                            result["response_time_ms"]
                        )
                
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected", session_id=session_id)
        except Exception as e:
            logger.error("WebSocket error", session_id=session_id, error=str(e))
        finally:
            # Cleanup websocket reference
            if session_id in session_manager.sessions:
                session_manager.sessions[session_id]["websocket"] = None

# =============================================================================
# STARTUP FUNCTIONS
# =============================================================================

def create_web_app() -> Optional[FastAPI]:
    """Creates the web application if dependencies are available"""
    if not all([FASTAPI_AVAILABLE, PYDANTIC_AVAILABLE, CHATBOT_AVAILABLE]):
        missing = []
        if not FASTAPI_AVAILABLE:
            missing.append("fastapi uvicorn")
        if not PYDANTIC_AVAILABLE:
            missing.append("pydantic")
        if not CHATBOT_AVAILABLE:
            missing.append("chatbot_integration module")
        
        print(f"❌ Web Interface nicht verfügbar. Fehlende Dependencies: {', '.join(missing)}")
        return None
    
    return app

def run_web_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Runs the web server"""
    if not create_web_app():
        return
    
    print(f"🌐 Starting BU-Processor Chatbot Web Interface...")
    print(f"   🔗 Web UI: http://{host}:{port}")
    print(f"   📚 API Docs: http://{host}:{port}/docs")
    print(f"   ⚡ WebSocket: ws://{host}:{port}/ws/{{session_id}}")
    
    try:
        uvicorn.run("chatbot_web_interface:app", host=host, port=port, reload=reload)
    except Exception as e:
        print(f"❌ Web Server Error: {e}")

if __name__ == "__main__":
    run_web_server()
