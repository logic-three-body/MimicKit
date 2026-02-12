# ASE

![ASE](../images/ASE_teaser.png)

"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters"
(https://xbpeng.github.io/projects/ASE/index.html).

---

To train an ASE model, use the following command:
```
python mimickit/run.py --mode train --num_envs 4096 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/ase_humanoid_sword_shield_env.yaml --agent_config data/agents/ase_humanoid_agent.yaml --visualize false --out_dir output/
```
To test an ASE model, run the following command:
```
python mimickit/run.py --mode test --num_envs 4 --engine_config data/engines/isaac_gym_engine.yaml --env_config data/envs/ase_humanoid_sword_shield_env.yaml --agent_config data/agents/ase_humanoid_agent.yaml --visualize true --model_file data/models/ase_humanoid_sword_shield_model.pt
```

## Paper-to-Code Notes
- Paper objective combines adversarial imitation and mutual-information regularization for skill latents.
- Core implementation entry points:
  - skill-conditioned policy: `mimickit/learning/ase_model.py:14`, `mimickit/learning/ase_model.py:20`
  - encoder reward/loss: `mimickit/learning/ase_agent.py:212`, `mimickit/learning/ase_agent.py:309`
  - latent scheduling: `mimickit/learning/ase_agent.py:95`, `mimickit/learning/ase_agent.py:116`
- Detailed formula-level mapping: [`docs/paper_code/README_ASE_TheoryCode.md`](paper_code/README_ASE_TheoryCode.md)

## Citation
```
@article{
	2022-TOG-ASE,
	author = {Peng, Xue Bin and Guo, Yunrong and Halper, Lina and Levine, Sergey and Fidler, Sanja},
	title = {ASE: Large-scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters},
	journal = {ACM Trans. Graph.},
	issue_date = {August 2022},
	volume = {41},
	number = {4},
	month = jul,
	year = {2022},
	articleno = {94},
	publisher = {ACM},
	address = {New York, NY, USA},
	keywords = {motion control, physics-based character animation, reinforcement learning}
}
```
