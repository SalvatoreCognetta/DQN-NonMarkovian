wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh

bash ~/miniconda.sh -u -b -p

rm ~/miniconda.sh

source $HOME/miniconda3/bin/activate

echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc 

printf '\n# add path to conda\nexport PATH="$HOME/miniconda3/bin:$PATH"\n' >> ~/.bashrc

source ~/.bashrc

conda init --help

source ~/.bashrc

conda activate

source ~/.bashrc

