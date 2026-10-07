# Third-party data and services

## BrowseComp-Plus

This repository downloads, but does not redistribute:

- [`Tevatron/browsecomp-plus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus)
- [`Tevatron/browsecomp-plus-corpus`](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus)

Both Hugging Face dataset cards identify the datasets as MIT-licensed. Follow
the current dataset cards and upstream repository for their terms. Keep
decrypted benchmark material out of source control and public artifacts. The
dataset-level license metadata should not be read as legal advice about rights
in every underlying web document; this repository downloads the corpus at
runtime and does not redistribute it.

If you use the benchmark, cite:

```bibtex
@article{chen2025BrowseCompPlus,
  title={BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent},
  author={Chen, Zijian and Ma, Xueguang and Zhuang, Shengyao and Nie, Ping and Zou, Kai and Liu, Andrew and Green, Joshua and Patel, Kshama and Meng, Ruoxi and Su, Mingyi and Sharifymoghaddam, Sahel and Li, Yanxi and Hong, Haoran and Shi, Xinyu and Liu, Xuye and Thakur, Nandan and Zhang, Crystina and Gao, Luyu and Chen, Wenhu and Lin, Jimmy},
  year={2025},
  journal={arXiv preprint arXiv:2508.06600}
}
```

Upstream project: <https://github.com/texttron/BrowseComp-Plus>

The decryption routine in `browsecomp_plus/eval/decrypt_eval.py` is derived
from the upstream project's published de-obfuscation example. Its MIT notice
is reproduced in
[`LICENSES/BrowseComp-Plus-MIT.txt`](LICENSES/BrowseComp-Plus-MIT.txt).

## External services

Running this eval uses the Bigdata Content and Search APIs and AWS Bedrock.
Their SDKs, APIs, model availability, pricing, quotas, and terms are provided
by their respective owners and are not included in this repository.
