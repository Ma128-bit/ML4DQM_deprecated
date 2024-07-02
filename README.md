# Training DQM Tool for CSC

```
git clone https://github.com/Ma128-bit/ML4DQM.git
```

## Get Monitoring elements (ME)
`Requires a working conda installation`
```=shell
conda create --name MEProd python=3.9
conda activate MEProd
pip3 install -r requirements_getME.txt 
chmod +x submit.sh
```
Use `Submit_getMEwithDIALS.py` to submit (with **condor**) the code, based on [dials_api](https://github.com/cms-DQM/dials-py), that gets the MEs. List of arguments:

| Argument                   | Default    | Required | Description                                |
| -------------------------- | :--------: | :------: | ------------------------------------------ |
| `-w / --workspace`         | csc        | False    | DIALS-workspace                            |
| `-m / --menames`           |            | True     | One or a list of monitoring elements       |
| `-d / --dtnames`           |            | False    | One or a list of dataset elements (if None takes all the possible datasets) |
| `-t / --metype`            |            | True     | Type of monitoring elements h1d or h2d     |
| `-o / --outputdir`         | test       | False    | Output directory                           |
| `-c / --conda`             | MEProd     | False    | Conda environment name                     |
| `-p / --miniconda_path`    |            | True     | Path to the miniconda installation directory |
| `--min_run`                |            | True     | Minimum run (Not required if `--era`)        |
| `--max_run`                |            | True     | Maximum run (Not required if `--era`)        |
| `--max_splits`             | 16         | False    | Number of splits per ME                      |
| `--era`                    |            | False    | Automatically select the min and max run according to the chosen era (ex: Run2024D)|

Usage example:
```
python3 Submit_getMEwithDIALS.py -m CSC/CSCOfflineMonitor/recHits/hRHGlobalm4 -t h2d -p /lustrehome/mbuonsante/miniconda3 \
-c cmsdials --era Run2024E --n_splits 20 --outputdir hRHGlobalm4E
```

To ensure that all the jobs have finished, use:
```=shell
grep "Done:" "outputdir"/log/*.out | wc -l
```
**Note:**

If you get the error:

`ImportError: cannot import name 'MutableMapping' from 'collections' `

Modify `classad/_expression.py` changing `from collections import MutableMapping` with `from collections.abc import MutableMapping`


## Fetch image info
**Note:**
If you run the notebook on SWAN there is no need to follow the steps below. You only need to install oms-api-client and runregistry_api_client and import them as:
```
import sys
sys.path.append('run registry site')
sys.path.append('./oms-api-client')
```
where `run registry site` is obtained usign: `pip show runregistry`


```=shell
conda create --name PrePro python=3.9
conda activate PrePro
pip3 install -r requirementsPrePro.txt 
```
Follow the "Authentication Prerequisites" instructions on [runregistry_api_client](https://github.com/cms-DQM/runregistry_api_client). Then follow [oms-api-client](https://gitlab.cern.ch/cmsoms/oms-api-client) instructions. (You can use the same application for both runregistry and oms)
Save the oms application credentials in a file named `config.yaml` with this structure:
```=yaml
APIClient:
    client_ID: 'id_example'
    Client_Secret: 'secret_example'
```
Run the notebook: `CSC_AE_getInfo.ipynb`

## Pre-processing
```=shell
conda activate PrePro
```
Run the notebook: `CSC_AE_preprocessing.ipynb`

## Train Autoencoder
Run the notebook: `CSC_AE_training.ipynb`
