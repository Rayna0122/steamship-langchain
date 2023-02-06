# Chatbot Example

This is an example of deploying an app using a LangChain-based ChatBot on Steamship.

## Try it out

Install dependencies:
```commandline
pip install steamship-langchain
pip install termcolor
```

Run the client:
```commandline
python3.8 client/client.py
```

## Deploy your own

Switch to the `server/` directory and run deploy. 
```commandline
cd server
ship deploy
```

The deployment script will walk you through setting up a package name that you 
can use for your own instance, if desired. This will enable you to modify the example
to meet your needs (or just to have fun experimenting with).

After deployment, switch back to the parent directory (`$ cd ..`) to run the client, etc.
You'll need to update the `package_handle` in the client to match your new deployment.

### api.py

`api.py` is required by Steamship for packages being deployed. You may add additional source
files as desired, but you MUST always have `api.py`.

### A note on dependencies

Steamship relies on `requirements.txt` as part of the packaging and deploy. If you add
new dependencies to your server code, please ensure they are reflected in `requirements.txt`.