"""Wrapper API: simplified high-level endpoints on top of the ComfyUI server.

The wrapper hides the internal node-graph details and model setup: callers
POST a prompt plus an input image and get back a job id, then poll the job
for the generated image. Missing models for the workflow are downloaded
automatically.
"""

__version__ = "0.1.0"


def register_wrapper_routes(routes, prompt_server):
    """Mount the wrapper API on the server's route table.

    ``routes`` is the PromptServer ``web.RouteTableDef``; every endpoint here
    is also exposed under the ``/api`` prefix by the server.
    """
    from api_wrapper.routes import register_wrapper_routes as _register

    return _register(routes, prompt_server)
