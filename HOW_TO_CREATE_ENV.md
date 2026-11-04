# How to Create Environment

## On your local machine

### Create a virtual environment:  
```sh
python3 -m venv .venv
```

### Activate the environment:  
```sh
source .venv/bin/activate
```

### Install dependencies:  
```sh
pip install -r requirements.txt
```

---

## On an HPC cluster

On a cluster the scientific packages (NumPy, SciPy, pandas, scikit-learn, Matplotlib, Seaborn, JupyterLab)
come from software modules that are already optimised for the hardware. The [env.sh](env.sh) script loads
those modules and adds a small virtual environment on top for the few packages that are missing, such as Marimo.

### First time  
```sh
git clone https://github.com/gozsari/ML-OneDay-Course.git ~/ML-OneDay-Course
cd ~/ML-OneDay-Course
source env.sh
```
This first run creates `my_venv` and installs the remaining packages, which takes a few minutes.

### Every new session  
```sh
source ~/ML-OneDay-Course/env.sh
```
That single line loads the modules and activates `my_venv`. Note that the script must be **sourced**, not executed.

If you cloned the repository somewhere else, point the script at it before sourcing:
```sh
export ML_COURSE_DIR=/path/to/your/clone
source $ML_COURSE_DIR/env.sh
```

### Running the notebooks on a compute node

Do not run the notebooks on the login node. Start an interactive job first:
```sh
srun -A <project> -p <partition> -t 02:00:00 --pty bash
cd ~/ML-OneDay-Course
source env.sh
```

Then start Jupyter:
```sh
jupyter lab --no-browser --ip=$(hostname) --port=8888
```
or the Marimo book:
```sh
cd marimo_book
marimo edit --headless --host $(hostname) --port 2718
```

Finally, open a tunnel from your own machine, using the node name printed in the job:
```sh
ssh -L 8888:<node>:8888 <user>@<login-node>
```
Open the URL printed by Jupyter or Marimo in your browser. Keep the access token in that URL,
since compute nodes are shared with other users.

### Troubleshooting

- To rebuild the environment from scratch: `rm -rf ~/ML-OneDay-Course/my_venv && source env.sh`
- `pip list --local` should show mainly Marimo. If NumPy or SciPy appear there, pip has installed its own
  copies over the module versions, so rebuild the environment.
- If the cluster updates its software stack, adjust the version numbers in the `module load` lines in
  [env.sh](env.sh) and rebuild.

The module names in [env.sh](env.sh) are those of the cluster this course was prepared for.
On another system, run `module spider <package>` to find the matching names.
