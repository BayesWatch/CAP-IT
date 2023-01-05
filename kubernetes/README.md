# Running MLProject experiments on kubernetes

1. Create a docker image for your experiments by:
   1. Modify the [Dockerfile](../Dockerfile) to build an image that can succesfully run your experiment.
   2. Build the images by running `docker build . -t image-name:version` from within the minimal-ml-template directory.
   3. Go to your [Github token generation page](https://github.com/settings/tokens/new) and generate a token that allows package management push permissions. Copy the new token.
   4. Then to push your image to the github image registry, run:
        ```bash
        export CR_PAT=YOUR_TOKEN ; echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
        docker tag SOURCE_IMAGE_NAME:VERSION ghcr.io/TARGET_OWNER/TARGET_IMAGE_NAME:VERSION
        docker push ghcr.io/OWNER/IMAGE_NAME:VERSION
        ```
        Replacing YOUR_TOKEN with your github api key, USERNAME with your github username, SOURCE_IMAGE_NAME:VERSION with the local name and version of the image you build, and TARGET_OWNER/TARGET_IMAGE_NAME:VERSION with the target github owner (TARGET_OWNER) under which you want to push the package, along with a package name (TARGET_IMAGE_NAME) and a version (VERSION)
 2. Once your image has been pushed to a relevant image registry, you should fill in the variables in the file [setup_variables.sh](setup_variables.sh) file with their respective values, and then export the variables to your local VM and the kubernetes cluster by running:
    ```bash
    source kubernetes/setup_variables.sh
    bash kubernetes/setup_kubernetes_variables.sh
    bash kubernetes/setup_secrets.sh
    ```
 3. Modify the [runner script](run_kube_experiments.py) to generate all the experiment commands you'd like to be launched in the kubernetes cluster by modifying the `get_scripts()` method.
 4. Run the runner script to launch your experiments:
    ```bash
    python kubernetes/run_kube_experiments.py
    ```
   
# Setting up your machine to work with a kubernetes cluster using bwatchcompute tools
bwatchcompute: A set of tools built to simplify daily driving of cloud resources for individual VM access, Kubernetes batch jobs and miscellaneous useful functionality related to cloud-based ML research

## Installation
To install the toolset, and get your environment ready to run Kubernetes jobs, you need to:
1. Log into a machine to be used as the job generator and submission terminal. We recommend that this is a google cloud VM with at least 4 CPU cores, or, your local machine -- although doing this on a google cloud VM generally has less probability of issues. 
2. Pull and run the controller docker by running:
   ```bash
   docker pull ghcr.io/bayeswatch/controller:0.1.0
   docker run -it ghcr.io/bayeswatch/controller:0.1.0
   ```
3. Clone repository to your local machine, or to a remote machine meant to be the job submission client

    ```
    git clone https://github.com/BayesWatch/bwatchcompute.git
    ```
4. If intenting to develop new features to push to the Github repository, you need to:
   1. Log into your github account
        ```bash
        gh auth login
        ```
    2. Set up your defaul email and name in github
        ```bash
        git config --global user.email "you@example.com"
        git config --global user.name "Your Name"
        ```
5. Sign in to your gcloud account:

    ```bash
    gcloud auth login
    ```

6. Select the gcloud project to be tali-multi-modal by running:

    ```bash
    gcloud config set project tali-multi-modal
    ```
7. Sign into the gpu kubernetes cluster
   ```bash
   gcloud container clusters get-credentials gpu-cluster-1 --zone us-central1-a --project tali-multi-modal
   ```
8. Set up the environment variables by filling the variables in `tutorial/setup_variables.sh` and then running:
   ```bash
   source tutorial/setup_variables.sh
   ```

## Cheatsheet
A list of commands to make your life easier when working with kubernetes
### Useful for kubectl 
Listing VM nodes of the cluster
```bash
kubectl get nodes
```

Listing pods of the cluster
```bash
kubectl get pods
```

Listing jobs of the cluster
```bash
kubectl get jobs
```

Read logs of a particular pod
```bash
kubectl logs <pod_id>
```

Read meta logs of a particular pod
```bash
kubectl describe pod <pod_id>
```

Submit a job to the cluster
```bash
kubectl create -f job.yaml
```

### Autocomplete kubectl

To enable autocomplete for kubectl:

#### Fish auto-completion
The kubectl completion script for Fish can be generated with the command kubectl completion fish. Sourcing the completion script in your shell enables kubectl autocompletion.

To do so in all your shell sessions, add the following line to your `~/.config/fish/config.fish` file:

```bash
kubectl completion fish | source
```

For other autocompletion tools see the [autocompletion documentation](https://kubernetes.io/docs/tasks/tools/included/)

#### Bash auto-completion
You now need to ensure that the kubectl completion script gets sourced in all your shell sessions. There are two ways in which you can do this:


```bash
echo 'source <(kubectl completion bash)' >>~/.bashrc
```

If you have an alias for kubectl, you can extend shell completion to work with that alias:

```bash
echo 'alias k=kubectl' >>~/.bashrc
echo 'complete -o default -F __start_kubectl k' >>~/.bashrc
```

Note: bash-completion sources all completion scripts in `/etc/bash_completion.d`.
Both approaches are equivalent. After reloading your shell, kubectl autocompletion should be working. To enable bash autocompletion in current session of shell, run exec bash:

```bash
exec bash
```

### Managing Kubernetes secrets

This section contains commands that help one configure secrets for kubernetes with the bear minimum of commands. For a more detailed description look at this [article](https://spacelift.io/blog/kubernetes-secrets).

Create a namespace to store your secrets in:

```bash
kubectl create namespace <namespace-name>
```

Store secrets using the following:

```bash
kubectl create secret generic <folder-for-secrets-name> \
    --from-literal=PASSWORD=password1234 \
    --namespace=<namespace-name>
```

To see the saved secrets use:

```bash
kubectl -n <namespace-name> get secret <folder-for-secrets-name> -o jsonpath='{.data.PASSWORD}' | base64 --decode
kubectl -n <namespace-name> describe secrets/<folder-for-secrets-name>
kubectl -n <namespace-name> get secrets
```

### Managing external GCP persistent disks with Kubernetes

See the documentation at https://cloud.google.com/kubernetes-engine/docs/concepts/persistent-volumes

the TLDR is, use the below as a guiding manifest for your disk claims and consumptions

```yaml
# pvc-pod-demo.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-demo
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 30Gi
  storageClassName: standard-rwo
---
kind: Pod
apiVersion: v1
metadata:
  name: pod-demo
spec:
  volumes:
    - name: pvc-demo-vol
      persistentVolumeClaim:
       claimName: pvc-demo
  containers:
    - name: pod-demo
      image: nginx
      resources:
        limits:
          cpu: 10m
          memory: 80Mi
        requests:
          cpu: 10m
          memory: 80Mi
      ports:
        - containerPort: 80
          name: "http-server"
      volumeMounts:
        - mountPath: "/usr/share/nginx/html"
          name: pvc-demo-vol
```
