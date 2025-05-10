from app.app import app
import uvicorn


if __name__ == "__main__":
    from app.config import settings

    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        debug=settings.APP_DEBUG,
    )
