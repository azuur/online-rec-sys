import uvicorn
from online_learning_demo.api import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
