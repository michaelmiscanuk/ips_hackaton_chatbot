"""
Flask Frontend for Text Analysis Application

This Flask application provides a beautiful web interface for the
LangGraph text analysis backend. It communicates with the FastAPI
backend via REST API calls.
"""

from flask import Flask, render_template, request, jsonify, flash
import requests
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "your-secret-key-change-in-production")

# Backend API configuration
# Use environment variable for production, localhost for development
API_BASE_URL = os.environ.get(
    "API_BASE_URL", "http://localhost:8000"  # Default to local backend
)

logger.info(f"Frontend initialized. Backend API: {API_BASE_URL}")


@app.route("/")
def index():
    """
    Home page - Main text analysis interface
    """
    return render_template("index.html")


@app.route("/about")
def about():
    """
    About page - Information about the application
    """
    return render_template("about.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """
    Proxy endpoint to backend Chat API
    """
    try:
        # Get form data
        data = request.get_json()
        message = data.get("message", "").strip()
        thread_id = data.get("thread_id")

        logger.info(f"Received chat request. Message length: {len(message)}")
        if thread_id:
            logger.info(f"Thread ID: {thread_id}")

        if not message:
            return jsonify({"success": False, "error": "Message cannot be empty"}), 400

        # Forward request to backend API
        backend_url = f"{API_BASE_URL}/api/chat"
        logger.info(f"Forwarding to backend: {backend_url}")

        response = requests.post(
            backend_url,
            json={"message": message, "thread_id": thread_id},
            timeout=120,
        )

        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            logger.info("Chat request completed successfully")
            return jsonify(result), 200
        else:
            try:
                error_detail = response.json().get("detail", "Unknown error")
            except:
                error_detail = f"Status code: {response.status_code}"

            logger.error(f"Backend error: {error_detail}")
            return (
                jsonify({"success": False, "error": f"Backend error: {error_detail}"}),
                response.status_code,
            )

    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to backend at {API_BASE_URL}")
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Cannot connect to backend API. Please make sure the backend is running.",
                }
            ),
            503,
        )

    except Exception as e:
        logger.error(f"Unexpected error in chat proxy: {str(e)}", exc_info=True)
        return (
            jsonify(
                {"success": False, "error": f"An unexpected error occurred: {str(e)}"}
            ),
            500,
        )


@app.route("/health")
def health():
    """
    Health check endpoint
    """
    try:
        # Check if backend is reachable
        backend_url = f"{API_BASE_URL}/health"
        response = requests.get(backend_url, timeout=5)
        backend_healthy = response.status_code == 200
    except:
        backend_healthy = False

    return jsonify(
        {
            "status": "healthy",
            "frontend": "operational",
            "backend": "operational" if backend_healthy else "unreachable",
            "backend_url": API_BASE_URL,
        }
    )


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template("404.html"), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal error: {str(error)}", exc_info=True)
    return render_template("500.html"), 500


def main():
    """
    Main entry point for running the Flask application
    """
    port = int(os.environ.get("PORT", 5000))
    app.run(
        host="0.0.0.0", port=port, debug=os.environ.get("FLASK_ENV") == "development"
    )


# For Vercel deployment
# Vercel will use this app object directly
if __name__ == "__main__":
    main()
