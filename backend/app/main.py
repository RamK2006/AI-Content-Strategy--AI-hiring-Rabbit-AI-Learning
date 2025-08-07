# Backend Main Application

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import asyncio
import json
import uvicorn
from datetime import datetime
from typing import List, Dict, Any

from app.config import settings
from app.database import init_db, get_db
from routers import strategy, content, influencer, analytics, trends
from services.ai_service import AIService
from services.trend_service import TrendService
from models.user import User

# WebSocket connection manager for real-time features
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast_metrics(self, data: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(data))
            except:
                self.active_connections.remove(connection)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database and services
    await init_db()
    
    # Start background tasks
    asyncio.create_task(real_time_metrics_publisher())
    asyncio.create_task(trend_monitoring_service())
    
    yield
    
    # Cleanup on shutdown
    print("Shutting down services...")

app = FastAPI(
    title="AI Influencer Content Strategy Platform",
    description="Complete platform for predictive content strategy and influencer marketing automation",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Include routers
app.include_router(strategy.router, prefix="/api/strategy", tags=["Strategy"])
app.include_router(content.router, prefix="/api/content", tags=["Content"])
app.include_router(influencer.router, prefix="/api/influencer", tags=["Influencer"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(trends.router, prefix="/api/trends", tags=["Trends"])

# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/real-time")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "subscribe_metrics":
                # Subscribe to specific metrics
                pass
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# Feature 1: Competitive Strategy Mimicry Engine
@app.get("/api/competitive-analysis/{competitor_domain}")
async def analyze_competitor(competitor_domain: str, db=Depends(get_db)):
    """Analyze competitor strategy and generate adaptation blueprint"""
    strategy_service = StrategyService()
    analysis = await strategy_service.analyze_competitor_strategy(competitor_domain)
    adaptation_blueprint = await strategy_service.generate_adaptation_blueprint(analysis)
    
    return {
        "competitor": competitor_domain,
        "strategy_analysis": analysis,
        "adaptation_blueprint": adaptation_blueprint,
        "roi_forecast": analysis.get("predicted_roi", 0),
        "implementation_timeline": "14-21 days"
    }

# Feature 2: Predictive Content Performance Modeling
@app.post("/api/content/predict-performance")
async def predict_content_performance(content_data: dict, db=Depends(get_db)):
    """Predict content performance before publication"""
    ai_service = AIService()
    prediction = await ai_service.predict_content_performance(content_data)
    
    return {
        "engagement_prediction": prediction["engagement"],
        "reach_prediction": prediction["reach"],
        "conversion_prediction": prediction["conversion"],
        "confidence_score": prediction["confidence"],
        "optimal_timing": prediction["optimal_time"],
        "format_recommendations": prediction["format_suggestions"]
    }

# Feature 3: Dynamic Strategy Roadmap Generator
@app.post("/api/strategy/generate-roadmap")
async def generate_strategy_roadmap(goals: dict, db=Depends(get_db)):
    """Generate adaptive strategy roadmap"""
    strategy_service = StrategyService()
    roadmap = await strategy_service.generate_dynamic_roadmap(goals)
    
    return {
        "roadmap": roadmap["plan"],
        "milestones": roadmap["milestones"],
        "budget_allocation": roadmap["budget"],
        "success_probability": roadmap["success_score"],
        "adaptation_triggers": roadmap["triggers"]
    }

# Feature 4: Cross-Platform Strategy Orchestration
@app.post("/api/content/cross-platform-orchestration")
async def orchestrate_cross_platform(campaign_data: dict, db=Depends(get_db)):
    """Orchestrate content across multiple platforms"""
    content_service = ContentService()
    orchestration = await content_service.orchestrate_platforms(campaign_data)
    
    return {
        "platform_strategies": orchestration["strategies"],
        "content_flow": orchestration["flow"],
        "publishing_schedule": orchestration["schedule"],
        "brand_consistency_score": orchestration["consistency"]
    }

# Feature 5: Trend Cascade Prediction System
@app.get("/api/trends/cascade-prediction/{trend_topic}")
async def predict_trend_cascade(trend_topic: str, db=Depends(get_db)):
    """Predict how trends will cascade across platforms"""
    trend_service = TrendService()
    cascade = await trend_service.predict_trend_cascade(trend_topic)
    
    return {
        "trend": trend_topic,
        "current_stage": cascade["stage"],
        "platform_migration": cascade["migration"],
        "optimal_entry_point": cascade["entry_timing"],
        "longevity_forecast": cascade["longevity"]
    }

# Feature 6: Strategic Gap Analysis
@app.post("/api/strategy/gap-analysis")
async def analyze_strategic_gaps(market_data: dict, db=Depends(get_db)):
    """Identify strategic opportunities and content gaps"""
    strategy_service = StrategyService()
    gaps = await strategy_service.analyze_strategic_gaps(market_data)
    
    return {
        "content_gaps": gaps["content"],
        "audience_opportunities": gaps["audience"],
        "blue_ocean_topics": gaps["blue_ocean"],
        "competitive_vulnerabilities": gaps["vulnerabilities"]
    }

# Feature 13: Ask Niche - What's Poppin' Feature
@app.get("/api/trends/whats-poppin/{niche}")
async def whats_poppin_in_niche(niche: str, db=Depends(get_db)):
    """Get instant insights on what's trending in specific niche"""
    trend_service = TrendService()
    trends = await trend_service.get_niche_trends(niche)
    
    return {
        "niche": niche,
        "top_micro_trends": trends["micro_trends"],
        "hashtags": trends["hashtags"],
        "content_formats": trends["formats"],
        "suggested_influencers": trends["influencers"],
        "growth_trajectory": trends["trajectory"]
    }

# Feature 14: Audience Sentiment Pulse
@app.get("/api/sentiment/pulse/{topic}")
async def get_sentiment_pulse(topic: str, db=Depends(get_db)):
    """Get live audience sentiment for content tonality guidance"""
    ai_service = AIService()
    sentiment = await ai_service.analyze_real_time_sentiment(topic)
    
    return {
        "topic": topic,
        "sentiment_score": sentiment["score"],
        "mood_indicators": sentiment["mood"],
        "tonality_suggestions": sentiment["tonality"],
        "emotional_context": sentiment["emotions"],
        "engagement_recommendations": sentiment["recommendations"]
    }

# Background task for real-time metrics
async def real_time_metrics_publisher():
    """Background task to publish real-time metrics"""
    while True:
        try:
            # Collect real-time metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "engagement_rate": 2.8,  # Mock data - replace with real data
                "reach": 45200,
                "sentiment_score": 0.75,
                "trending_topics": ["AI", "Sustainability", "Web3"],
                "competitor_activity": {"Brand A": 15.2, "Brand B": -3.4}
            }
            
            await manager.broadcast_metrics(metrics)
            await asyncio.sleep(5)  # Update every 5 seconds
            
        except Exception as e:
            print(f"Error in metrics publisher: {e}")
            await asyncio.sleep(10)

# Background task for trend monitoring
async def trend_monitoring_service():
    """Background service for continuous trend monitoring"""
    while True:
        try:
            trend_service = TrendService()
            await trend_service.monitor_emerging_trends()
            await asyncio.sleep(60)  # Check every minute
            
        except Exception as e:
            print(f"Error in trend monitoring: {e}")
            await asyncio.sleep(120)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )