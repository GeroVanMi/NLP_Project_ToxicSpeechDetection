from visualization.model_server import app

if __name__ == '__main__':
    app.run("0.0.0.0", 5095, debug=False)
