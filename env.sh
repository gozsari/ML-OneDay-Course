# Source this file, do not execute it:  source env.sh
# The first run creates my_venv and installs marimo; later runs just activate it.
# Override the location of the clone with: export ML_COURSE_DIR=/path/to/clone

ML_COURSE_DIR="${ML_COURSE_DIR:-$HOME/ML-OneDay-Course}"
VENV_DIR="$ML_COURSE_DIR/my_venv"

module purge
module load virtualenv/20.32.0-GCCcore-14.3.0 \
            SciPy-bundle/2025.07-gfbf-2025b \
            scikit-learn/1.7.1-gfbf-2025b \
            matplotlib/3.10.5-gfbf-2025b \
            Seaborn/0.13.2-gfbf-2025b \
            JupyterLab/4.4.9-GCCcore-14.3.0

if [ ! -d "$VENV_DIR" ]; then
    echo "First run: creating $VENV_DIR ..."
    # jupyter and notebook come from the JupyterLab module, so they are filtered out here
    virtualenv --system-site-packages "$VENV_DIR" && source "$VENV_DIR/bin/activate" &&
    grep -vE '^(jupyter|notebook)' "$ML_COURSE_DIR/requirements.txt" \
        | pip install --no-cache-dir -r /dev/stdin || {
        echo "Environment setup failed; removing $VENV_DIR"
        deactivate 2>/dev/null
        rm -rf "$VENV_DIR"
        return 1
    }
    echo "Environment ready. Next time just run: source env.sh"
else
    source "$VENV_DIR/bin/activate"
fi
