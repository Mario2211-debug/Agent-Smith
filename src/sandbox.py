import docker


def execute_sandbox(python_code: str):

    client = docker.from_env()

    container = client.containers.run(
        "python:3.13-slim",
        command="sleep 3600",
        detach=True,

        network_disabled=True,
        mem_limit="128m",
        # nano_cpus=500000000
    )

    try:
        command = ["python3", "-c", python_code]
        exit_code, output = container.exec_run(command,
                                               stream=False, demux=False)
        text_result = output.decode('utf-8')
        print(f"Exit code: {exit_code}")
        print("---- Output da sandbox (Docker) ----")
        print(text_result)
        print("----------------")
        return text_result
    finally:
        container.stop()
        container.remove()


code = "import sys; print('A correr dentro do Docker!'); print(f'Python version: {sys.version}')"
execute_sandbox("import sys; print('A correr dentro do Docker!'); print(f'Python version: {sys.version}')")
