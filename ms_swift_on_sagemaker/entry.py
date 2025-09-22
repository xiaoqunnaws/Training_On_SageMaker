import os
import json
import socket
import yaml
import sys

# import sagemaker_ssh_helper
# sagemaker_ssh_helper.setup_and_start_ssh()

def run_command(command, description=""):
    """执行命令并检查返回值"""
    print(f"Executing: {description if description else command}")
    result = os.system(command)
    if result != 0:
        print(f"ERROR: Command failed with exit code {result}")
        print(f"Failed command: {command}")
        sys.exit(1)
    return result

if __name__ == "__main__":

    hosts = json.loads(os.environ['SM_HOSTS'])
    current_host = os.environ['SM_CURRENT_HOST']
    host_rank = int(hosts.index(current_host))

    #Parse the IP address of the master node in the multiple nodes cluster of SageMaker training.
    master = json.loads(os.environ['SM_TRAINING_ENV'])['master_hostname']
    master_addr = socket.gethostbyname(master)

    os.environ['DS_BUILD_FUSED_ADAM'] = '1'
    os.environ['NODE_INDEX'] = str(host_rank)
    os.environ['SM_MASTER'] = str(master)
    os.environ['SM_MASTER_ADDR'] = str(master_addr)
    os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'

    # backend env config
    os.environ['FI_PROVIDER'] = 'efa'
    os.environ['NCCL_PROTO'] = 'simple'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['HCCL_OVER_OFI'] = '1'

    # 设置脚本权限
    run_command("chmod +x ./train_script_sagemaker.sh", "Setting permissions for train_script_sagemaker.sh")
    # run_command("chmod +x ./train_multi_node.sh", "Setting permissions for train_multi_node.sh")
    run_command("chmod +x ./s5cmd", "Setting permissions for s5cmd")

    print("*****************start cp foundation model*****************************")
    run_command("./s5cmd sync {0}/* {1}".format(os.environ['MODEL_S3_PATH'], os.environ["MODEL_LOCAL_PATH"]), 
                "Copying foundation model from S3")
    print(f'-----finished cp-------')

    # 执行训练脚本 - 这里是关键部分
    print("*****************start training*****************************")
    # run_command("/bin/bash -c ./train_multi_node.sh", "Running training script")
    # run_command("sleep 2h")
    run_command("/bin/bash -c ./train_script_sagemaker.sh", "Running training script")
    print("*****************training completed successfully*****************************")

    # print("*****************finished training, start cp finetuned model*****************************")
    run_command("ls -l /opt/ml/checkpoints", "Listing checkpoints")


    # Copy files instead of moving them
#     run_command(r'find /opt/ml/checkpoints -maxdepth 1 -type f ! -name "*.sagemaker-uploaded" ! -name "checkpoint-*" -print', 
#                 "Finding files to copy")
#     run_command(r'find /opt/ml/checkpoints -maxdepth 1 -type d ! -name "." ! -name ".." ! -name "checkpoint-*" -print', 
#                 "Finding directories to copy")

#     # Use cp -r for directories and cp for files
#     run_command(r'find /opt/ml/checkpoints -maxdepth 1 -type f ! -name "*.sagemaker-uploaded" ! -name "checkpoint-*" -exec cp {} "/opt/ml/model/" \;', 
#                 "Copying files to model directory")
#     run_command(r'find /opt/ml/checkpoints -maxdepth 1 -type d ! -name "." ! -name ".." ! -name "checkpoint-*" -exec cp -r {} "/opt/ml/model/" \;', 
#                 "Copying directories to model directory")
    print(f'-----finished ———')

