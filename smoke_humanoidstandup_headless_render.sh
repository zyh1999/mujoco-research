#!/bin/bash
#SBATCH -J humstand_render_smoke
#SBATCH -p interactive
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:05:00

set -u

PYTHON=/scratch/h99859yz/rat_default_shared_csf3/venvs/rat_public_default/bin/python
XML=/scratch/h99859yz/trust-region-main/assets/humanoidstandup.xml
if test ! -f "${XML}"; then
  XML=$(${PYTHON} -c 'import gymnasium; from pathlib import Path; print(Path(gymnasium.__spec__.origin).parent / "envs" / "mujoco" / "assets" / "humanoidstandup.xml")')
fi

for backend in egl osmesa; do
  echo "backend=${backend}"
  MUJOCO_GL="${backend}" "${PYTHON}" -c 'import mujoco,sys; model=mujoco.MjModel.from_xml_path(sys.argv[1]); renderer=mujoco.Renderer(model, height=64, width=64); renderer.update_scene(mujoco.MjData(model)); print(renderer.render().shape); renderer.close()' "${XML}"
  echo "rc=$?"
done
