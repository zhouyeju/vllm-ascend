import os
import requests
import signal
import subprocess
import time

WORKSPACE_DIR = "./tests/e2e/pd_disaggreate/chariot/"
RUN_INSTANCES_SCRIPT = os.path.join(WORKSPACE_DIR, "run_pd_with_chariot_connector.sh")
RUN_PROXY_SERVER_SCRIPT = os.path.join(WORKSPACE_DIR, "run_proxy_server.sh")
RUN_CHARIOT_SCRIPT = os.path.join(WORKSPACE_DIR, "run_chariot.sh")
CLEAN_CHARIOT_SCRIPT = os.path.join(WORKSPACE_DIR, "clean_chariot.sh")
HOST_IP = "127.0.0.1"
PROXY_PORT = 8000
PREFILL_PORT = 8100
DECODE_PORT = 8200
WORKER_PORT = 31530
ETCD_PORT = 2411
MODEL_NAME = "Qwen/Qwen2.5-7B"
PROMPT_ANSWER = {
    "who is the president of the united states?": "?\nDonald Trump"
}
RUN_INSTANCE_KEYWORDS = "vllm serve"
RUN_PROXY_SERVER_KEYWORDS = "simple_pd_proxy_server.py"


def start_chariot():
    proc = subprocess.Popen(["bash", RUN_CHARIOT_SCRIPT, f"{HOST_IP}", f"{WORKER_PORT}", f"{ETCD_PORT}"])


def clean_chariot():
    proc = subprocess.Popen(["bash", CLEAN_CHARIOT_SCRIPT, f"{HOST_IP}", f"{WORKER_PORT}"])


def start_instances():
    proc = subprocess.Popen(["bash", RUN_INSTANCES_SCRIPT, f"{MODEL_NAME}", f"{HOST_IP}", f"{PREFILL_PORT}", f"{DECODE_PORT}"])


def start_proxy_server():
    proc = subprocess.Popen(["bash", RUN_PROXY_SERVER_SCRIPT, f"{HOST_IP}", f"{PROXY_PORT}", f"{PREFILL_PORT}", f"{DECODE_PORT}"])


def clean_instances_and_proxy_server():
    instance_pids = get_pids_by_keyword(RUN_INSTANCE_KEYWORDS)
    proxy_pids = get_pids_by_keyword(RUN_PROXY_SERVER_KEYWORDS)
    for pid in proxy_pids + instance_pids:
        os.kill(pid, sig)
        except ProcessLookupError:
            print(f"No such process with PID {pid}")
        except PermissionError:
            print(f"Permission denied to send signal to PID {pid}")
        except Exception as e:
            print(f"Error: {e}")
        time.sleep(5)
        os.(pid, signal.SIGKILL)


def send_post_request(url, data):
    try:
        response = requests.post(url, json=data, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        return f"Request failed: {e}"
    

def get_pids_by_keyword(keyword):
    try:
        # Run 'ps aux' to get all running processes
        result = subprocess.run(['ps', 'aux'], stdout=subprocess.PIPE, text=True)
        lines = result.stdout.strip().split('\n')

        matching_pids = []

        for line in lines[1:]:  # Skip the header line
            if keyword in line:
                parts = line.split()
                pid = parts[1]  # PID is the second column
                matching_pids.append(pid)

        return matching_pids



def test_chariot_pd_dist():
    start_chariot()
    start_instances()
    start_proxy_server()
    proxy_url = = f"http://{HOST_IP}:{PROXY_PORT}/v1/competions"
    start_instances()
    start_proxy_server()
    for prompt, answer in PROMPT_ANSWER.items():
        data = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": 50,
            "temperature": 0
        }
        response = send_post_request(proxy_url, data)
        assert response == answer, f"wrong response: {response}, expected: {answer}"
    clean_instances_and_proxy_server()
    clean_chariot()
