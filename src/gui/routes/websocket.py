"""
WebSocket routes for real-time log streaming.

Provides WebSocket endpoints for streaming pipeline stage
execution logs to the browser in real-time.
"""
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.pipeline_service import get_pipeline_service

router = APIRouter()


@router.websocket("/ws/logs/{stage_name}")
async def stream_logs(websocket: WebSocket, stage_name: str):
    """
    Stream stage execution logs via WebSocket.

    The client connects, then the server runs the stage and
    streams each line of output as it becomes available.
    """
    await websocket.accept()

    service = get_pipeline_service()
    stage = service.get_stage(stage_name)

    if not stage:
        await websocket.send_json({
            "type": "error",
            "message": f"Stage '{stage_name}' not found"
        })
        await websocket.close()
        return

    try:
        # Send start message
        await websocket.send_json({
            "type": "start",
            "stage": stage_name,
            "message": f"Starting {stage_name}..."
        })

        # Stream logs from stage execution
        async for line in service.run_stage(stage_name):
            await websocket.send_json({
                "type": "log",
                "line": line
            })

        # Send completion message
        await websocket.send_json({
            "type": "complete",
            "stage": stage_name,
            "message": f"Stage {stage_name} finished"
        })

    except WebSocketDisconnect:
        # Client disconnected, that's okay
        pass
    except Exception as e:
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@router.websocket("/ws/pipeline")
async def pipeline_status(websocket: WebSocket):
    """
    Stream pipeline status updates.

    Sends periodic updates about stage status for dashboard refresh.
    """
    await websocket.accept()

    service = get_pipeline_service()

    try:
        while True:
            stages = service.discover_stages()
            await websocket.send_json({
                "type": "status",
                "stages": [
                    {
                        "name": s.name,
                        "status": s.status.value,
                        "has_qa_report": s.has_qa_report,
                    }
                    for s in stages
                ]
            })
            # Update every 5 seconds
            await asyncio.sleep(5)

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
