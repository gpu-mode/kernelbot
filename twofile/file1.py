# file1.py
from modal import App, Mount

mount = Mount.from_local_dir(".", remote_path="/root")
app = App("my-app")

@app.function(mounts=[mount])
def run_main():
    from file2 import main
    main()

if __name__ == "__main__":
    app.run()