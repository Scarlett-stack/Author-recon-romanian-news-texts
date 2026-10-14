# GHID RULARE CLUSTER FEP (poor bastards edition)

Nu vrei sa dai 11$ pe collab? No problemo avem solutii

### Instalare Apptainer

Nu te deranja sa rulezi build la imaginea de apptainer pe cluster o sa dea OOM. Apptainer trebuie instalat local si buil-uit tot acolo

```bash
sudo apt update

# instalam prerequisites
sudo apt install -y build-essential libseccomp-dev pkg-config squashfs-tools cryptsetup curl

# ne trebuie si go , versiune >= 20
go version

 export VERSION=1.11 OS=linux ARCH=amd64 && \
    wget https://dl.google.com/go/go$VERSION.$OS-$ARCH.tar.gz && \
    sudo tar -C /usr/local -xzvf go$VERSION.$OS-$ARCH.tar.gz && \
    rm go$VERSION.$OS-$ARCH.tar.gz

sudo apt remove -y golang-go # in caz ca se plange apptainer la versiune

sudo rm -rf /usr/local/go
wget https://go.dev/dl/go1.21.6.linux-amd64.tar.gz

sudo tar -C /usr/local -xzf go1.21.6.linux-amd64.tar.gz
echo 'export PATH=/usr/local/go/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# apptainer
wget https://github.com/apptainer/apptainer/releases/download/v1.3.0/apptainer-1.3.0.tar.gz
tar -xzf apptainer-1.3.0.tar.gz
cd apptainer..
./mconfig
sudo make install 

# configuram def :
nano transformers.def

Bootstrap: docker
From: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

%post
    pip install --no-cache-dir \
        transformers==4.41.2 \
        pandas \
        numpy \
        matplotlib \
        seaborn \
        scikit-learn

%environment
    export PYTHONUNBUFFERED=1

%runscript
    python Transformer_baseline.py


apptainer build --fakeroot transformers.sif transformers.def 

```

Pe fep cream un script slurm  in care alocam resurse, specificam nodul si comanda de run 

```
[daria_catalina.banu@fep10 ~]$ cat run_gpu.slurm 
#!/bin/bash
#SBATCH --job-name=bert_authorship
#SBATCH --partition=xl
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --mem=16G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

echo "Running on $(hostname)"
echo "Starting job..."

cd ~/project

apptainer exec --nv ~/transformers.sif python Transformer_baseline.py

echo "Job finished"
```

