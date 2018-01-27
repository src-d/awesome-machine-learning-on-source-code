# Awesome Machine Learning On Source Code [![Awesome Machine Learning On Source Code](https://awesome.re/badge.svg)](https://github.com/src-d/awesome-machine-learning-on-source-code)

A curated list of awesome machine learning frameworks and algorithms that work on top of source code. Inspired by [Awesome Machine Learning](https://github.com/src-d/awesome-machine-learning-on-code).

## Contents

<!-- MarkdownTOC depth=4 -->
- [Digests](#digests)
- [Articles](#articles)
    - [Papers](#papers)
    - [Posts](#posts)
    - [Talks](#talks)
- [Software](#software)
    - [Machine Learning](#software-ML)
    - [Utilities](#software-utilities)
- [Datasets](#datasets)
- [Credits](#credits)
- [Contributions](#contributions)
- [License](#license)

<!-- /MarkdownTOC -->

<a name="digests"></a>
## Digests
* [Learning from "Big Code"](http://learnbigcode.github.io)
* [A Survey of Machine Learning for Big Code and Naturalness](https://ml4code.github.io/)

<a name="articles"></a>
## Articles

<a name="papers"></a>
#### Papers 

* [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526v1) - Daniel A. Abolafia, Mohammad Norouzi, Quoc V. Le.
* [Code Completion with Neural Attention and Pointer Networks](https://arxiv.org/abs/1711.09573v1) - Jian Li, Yue Wang, Irwin King, Michael R. Lyu  .
* [Learning to Represent Programs with Graphs](https://arxiv.org/abs/1711.00740v1) - Miltiadis Allamanis, Marc Brockschmidt, Mahmoud Khademi.
* [Semantic Code Repair using Neuro-Symbolic Transformation Networks](https://arxiv.org/abs/1710.11054v1) - Jacob Devlin, Jonathan Uesato, Rishabh Singh, Pushmeet Kohli.
* [Neural Program Meta-Induction](https://arxiv.org/abs/1710.04157v1) - Jacob Devlin, Rudy Bunel, Rishabh Singh, Matthew Hausknecht, Pushmeet Kohli.
* [Code Attention: Translating Code to Comments by Exploiting Domain Features](https://arxiv.org/abs/1709.07642v2) - Wenhao Zheng, Hong-Yu Zhou, Ming Li, Jianxin Wu.
* [A Survey of Machine Learning for Big Code and Naturalness](https://arxiv.org/abs/1709.06182v1) - Miltiadis Allamanis, Earl T. Barr, Premkumar Devanbu, Charles Sutton.
* [Glass-Box Program Synthesis: A Machine Learning Approach](https://arxiv.org/abs/1709.08669v1) - Konstantina Christakopoulou, Adam Tauman Kalai.
* [Automatically Generating Commit Messages from Diffs using Neural Machine Translation](https://arxiv.org/abs/1708.09492v1) - Siyuan Jiang, Ameer Armaly, Collin McMillan.
* [A Parallel Corpus of Python Functions and Documentation Strings for Automated Code Documentation and Code Generation](https://arxiv.org/abs/1707.02275v1) - Antonio Valerio Miceli Barone, Rico Sennrich.
* [SmartPaste: Learning to Adapt Source Code](https://arxiv.org/abs/1705.07867) - Miltiadis Allamanis, Marc Brockschmidt.
* [Topic modeling of public repositories at scale using names in source code](https://arxiv.org/abs/1704.00135)
* [A Neural Architecture for Generating Natural Language Descriptions from Source Code Changes](https://arxiv.org/abs/1704.04856v1) - Pablo Loyola, Edison Marrese-Taylor, Yutaka Matsuo.
* [RobustFill: Neural Program Learning under Noisy I/O](https://arxiv.org/abs/1703.07469v1) - Jacob Devlin, Jonathan Uesato, Surya Bhupatiraju, Rishabh Singh, Abdel-rahman Mohamed, Pushmeet Kohli.
* [Neural Programming by Example](https://arxiv.org/abs/1703.04990v1) - Chengxun Shu, Hongyu Zhang.
* [Parameter-Free Probabilistic API Mining across GitHub](http://homepages.inf.ed.ac.uk/csutton/publications/fse2016.pdf)
* [A Subsequence Interleaving Model for Sequential Pattern Mining](http://homepages.inf.ed.ac.uk/csutton/publications/kdd2016-subsequence-interleaving.pdf)
* [Deep API Learning](https://arxiv.org/abs/1605.08535v3) - Xiaodong Gu, Hongyu Zhang, Dongmei Zhang, Sunghun Kim.
* [A Convolutional Attention Network for Extreme Summarization of Source Code](http://arxiv.org/abs/1602.03001)
* [Tailored Mutants Fit Bugs Better](https://arxiv.org/abs/1611.02516)
* [A deep language model for software code](https://arxiv.org/abs/1608.02715v1) - Hoa Khanh Dam, Truyen Tran, Trang Pham.
* [TASSAL: Autofolding for Source Code Summarization](http://homepages.inf.ed.ac.uk/csutton/publications/icse2016-demo.pdf)
* [Suggesting Accurate Method and Class Names](http://homepages.inf.ed.ac.uk/csutton/publications/accurate-method-and-class.pdf)
* [Mining idioms from source code](http://homepages.inf.ed.ac.uk/csutton/publications/idioms.pdf)
* [Mining Source Code Repositories at Massive Scale using Language Modeling](http://homepages.inf.ed.ac.uk/csutton/publications/msr2013.pdf)
* [Why, When, and What: Analyzing Stack Overflow Questions by Topic, Type, and Code](http://homepages.inf.ed.ac.uk/csutton/publications/msrCh2013.pdf)
* [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744) - Wang Ling, Edward Grefenstette, Karl Moritz Hermann, Tomáš Kočiský, Andrew Senior, Fumin Wang, Phil Blunsom.
* [Code Completion with Statistical Language Models](http://www.srl.inf.ethz.ch/papers/pldi14-statistical.pdf) - Veselin Raychev, Martin Vechev, Eran Yahav.
* [Using recurrent neural networks to predict next tokens in the java solutions](http://near.ai/articles/2017-06-01-Code-Completion-Demo/) - Alex Skidanov, Illia Polosukhin.
* [Learning Python Code Suggestion with a Sparse Pointer Network](https://arxiv.org/abs/1611.08307) - Avishkar Bhoopchand, Tim Rocktäschel, Earl Barr, Sebastian Riedel.
* [Learning Efficient Algorithms with Hierarchical Attentive Memory](http://arxiv.org/abs/1602.03218) - Andrychowicz, Marcin, and Karol Kurach.
* [DeepCoder: Learning to Write Programs](http://arxiv.org/abs/1611.01989) - Balog, Matej, Alexander L. Gaunt, Marc Brockschmidt, Sebastian Nowozin, and Daniel Tarlow.
* [Programming with a Differentiable Forth Interpreter](http://arxiv.org/abs/1605.06640) - Bošnjak, Matko, Tim Rocktäschel, Jason Naradowsky, and Sebastian Riedel.
* [Learning to Superoptimize Programs - Workshop Version](http://arxiv.org/abs/1612.01094) - Bunel, Rudy, Alban Desmaison, M. Pawan Kumar, Philip H. S. Torr, and Pushmeet Kohli.
* [Meta-Interpretive Learning of Efficient Logic Programs](https://uclmr.github.io/nampi/extended_abstracts/cropper.pdf) - Cropper, Andrew, and Stephen H. Muggleton.
* [Learning Operations on a Stack with Neural Turing Machines](http://arxiv.org/abs/1612.00827) - Deleu, Tristan, and Joseph Dureau.
* [Neural Functional Programming](http://arxiv.org/abs/1611.01988) - Feser, John K., Marc Brockschmidt, Alexander L. Gaunt, and Daniel Tarlow.
* [TerpreT: A Probabilistic Programming Language for Program Induction](http://arxiv.org/abs/1612.00817) - Gaunt, Alexander L., Marc Brockschmidt, Rishabh Singh, Nate Kushman, Pushmeet Kohli, Jonathan Taylor, and Daniel Tarlow.
* [Neural Turing Machines](http://arxiv.org/abs/1410.5401) - Graves, Alex, Greg Wayne, and Ivo Danihelka.
* [Neural Symbolic Machines: Learning Semantic Parsers on Freebase with Weak Supervision (Short Version)](http://arxiv.org/abs/1612.01197) - Liang, Chen, Jonathan Berant, Quoc Le, Kenneth D. Forbus, and Ni Lao.
* [Probabilistic Neural Programs](http://arxiv.org/abs/1612.00712) - Murray, Kenton W., and Jayant Krishnamurthy.
* [Neural Programmer: Inducing Latent Programs with Gradient Descent](http://arxiv.org/abs/1511.04834) - Neelakantan, Arvind, Quoc V. Le, and Ilya Sutskever.
* [Divide and Conquer with Neural Networks](http://arxiv.org/abs/1611.02401) - Nowak, Alex, and Joan Bruna.
* [Neural Programmer-Interpreters](http://arxiv.org/abs/1511.06279) - Reed, Scott, and Nando de Freitas.
* [Programs as Black-Box Explanations](http://arxiv.org/abs/1611.07579) - Singh, Sameer, Marco Tulio Ribeiro, and Carlos Guestrin.
* [A Differentiable Approach to Inductive Logic Programming](https://uclmr.github.io/nampi/extended_abstracts/yang.pdf) - Yang, Fan, Zhilin Yang, and William W. Cohen.
* [From Machine Learning to Machine Reasoning](http://arxiv.org/abs/1102.1808) - Bottou, Leon.
* [Learning Latent Multiscale Structure Using Recurrent Neural Networks](https://uclmr.github.io/nampi/extended_abstracts/chung.pdf) - Chung, Junyoung, Sungjin Ahn, and Yoshua Bengio.
* [Lifelong Perceptual Programming By Example](http://arxiv.org/abs/1611.02109) - Gaunt, Alexander L., Marc Brockschmidt, Nate Kushman, and Daniel Tarlow.
* [Inferring Algorithmic Patterns with Stack-Augmented Recurrent Nets](http://arxiv.org/abs/1503.01007) - Joulin, Armand, and Tomas Mikolov.
* [Neural GPUs Learn Algorithms](http://arxiv.org/abs/1511.08228) - Kaiser, Łukasz, and Ilya Sutskever.
* [API usage pattern recommendation for software development](http://www.sciencedirect.com/science/article/pii/S0164121216301200) - Haoran Niu, Iman Keivanloo, Ying Zou.
* [Summarizing Source Code using a Neural Attention Model](https://github.com/sriniiyer/codenn/blob/master/summarizing_source_code.pdf) University of Washington CSE, Seatle, WA, USA.
* [Program Synthesis from Natural Language Using Recurrent Neural Networks](https://homes.cs.washington.edu/~mernst/pubs/nl-command-tr170301.pdf) - University of Washington CSE, Seatle, WA, USA.
* [Exploring API Embedding for API Usages and Applications](http://home.eng.iastate.edu/~trong/projects/jv2cs/) - Nguyen, Nguyen, Phan and Nguyen.
* [Neural Nets Can Learn Function Type Signatures From Binaries](https://www.usenix.org/system/files/conference/usenixsecurity17/sec17-chua.pdf) - Zheng Leong Chua, Shiqi Shen, Prateek Saxena, and Zhenkai Liang.
* [Deep Learning Code Fragments for Code Clone Detection](http://www.cs.wm.edu/~denys/pubs/ASE'16-DeepLearningClones.pdf) -  Martin White, Michele Tufano, Christopher Vendome, and Denys Poshyvanyk.
* [Automated Identification of Security Issues from Commit Messages and Bug Reports](https://dl.acm.org/citation.cfm?id=3117771) [[PDF](http://asankhaya.github.io/pdf/automated-identification-of-security-issues-from-commit-messages-and-bug-reports.pdf)] - Yaqin Zhou and Asankhaya Sharma.
* [Neural Sketch Learning for Conditional Program Generation](https://arxiv.org/abs/1703.05698) - Vijayaraghavan Murali, Letao Qi, Swarat Chaudhuri, Chris Jermaine.

<a name="posts"></a>
#### Posts
* [Weighted MinHash on GPU helps to find duplicate GitHub repositories.](https://blog.sourced.tech//post/minhashcuda/)
* [Source Code Identifier Embeddings](https://blog.sourced.tech/post/id2vec/)
* [The half-life of code & the ship of Theseus](https://erikbern.com/2016/12/05/the-half-life-of-code.html)
* [The eigenvector of "Why we moved from language X to language Y"](https://erikbern.com/2017/03/15/the-eigenvector-of-why-we-moved-from-language-x-to-language-y.html)
* [Analyzing Github, How Developers Change Programming Languages Over Time](https://blog.sourced.tech/post/language_migrations/)

<a name="talks"></a>
#### Talks
* [Topic Modeling of GitHub Repositories](https://blog.sourced.tech//post/github_topic_modeling/)
* [Similarity of GitHub Repositories by Source Code Identifiers](http://vmarkovtsev.github.io/techtalks-2017-moscow/)
* [Using deep RNN to model source code](http://vmarkovtsev.github.io/re-work-2016-london/)
* [Source code abstracts classification using CNN (1)](http://vmarkovtsev.github.io/re-work-2016-berlin/)
* [Source code abstracts classification using CNN (2)](http://vmarkovtsev.github.io/data-natives-2016/)
* [Source code abstracts classification using CNN (3)](http://vmarkovtsev.github.io/slush-2016/)
* [Embedding the GitHub contribution graph](https://egorbu.github.io/techtalks-2017-moscow)

<a name="software"></a>
## Software

<a name="software-ML"></a>
#### Machine Learning
* [Differentiable Neural Computer (DNC)](https://github.com/deepmind/dnc) - TensorFlow implementation of the Differentiable Neural Computer.
* [sourced.ml](https://github.com/src-d/ml) - Abstracts feature extraction from source code syntax trees and working with ML models.
* [vecino](https://github.com/src-d/vecino) - Finds similar Git repositories.
* [apollo](https://github.com/src-d/apollo) - Source code deduplication as scale, research.
* [gemini](https://github.com/src-d/gemini) - Source code deduplication as scale, production.
* [enry](https://github.com/src-d/enry) - Insanely fast file based programming language detector.
* [Naturalize](https://github.com/mast-group/naturalize) - Language agnostic framework for learning coding conventions from a codebase and then expoiting this information for suggesting better identifier names and formatting changes in the code. 
* [Extreme Source Code Summarization](https://github.com/mast-group/convolutional-attention ) - Convolutional attention neural network that learns to summarize source code into a short method name-like summary by just looking at the source code tokens. 
* [Summarizing Source Code using a Neural Attention Model](https://github.com/sriniiyer/codenn) - CODE-NN, uses LSTM networks with attention to produce sentences that describe C# code snippets and SQL queries from StackOverflow. Torch over C#/SQL
* [Probabilistic API Miner](https://github.com/mast-group/api-mining) - Near parameter-free probabilistic algorithm for mining the most interesting API patterns from a list of API call sequences. 
* [Interesting Sequence Miner](https://github.com/mast-group/sequence-mining) - Novel algorithm that mines the most interesting sequences under a probabilistic model. It is able to efficiently infer interesting sequences directly from the database. 
* [TASSAL](https://github.com/mast-group/tassal) - Tool for the automatic summarization of source code using autofolding. Autofolding automatically creates a summary of a source code file by folding non-essential code and comment blocks.
* [JNice2Predict](http://www.nice2predict.org/) - Efficient and scalable open-source framework for structured prediction, enabling one to build new statistical engines more quickly.

<a name="software-utilities"></a>
#### Utilities
* [go-git](https://github.com/src-d/go-git) - Highly extensible Git implementation in pure Go which is friendly to data mining.
* [hercules](https://github.com/src-d/hercules) - Git repository mining framework with batteries on top of go-git.
* [bblfsh](https://github.com/bblfsh) - Self-hosted server for source code parsing.
* [engine](https://github.com/src-d/engine) - Scalable and distributed data retrieval pipeline for source code.
* [minhashcuda](https://github.com/src-d/minhashcuda) - Weighted MinHash implementation on CUDA to efficiently find duplicates.
* [kmcuda](https://github.com/src-d/kmcuda) - k-means on CUDA to cluster and to search for nearest neighbors in dense space.
* [wmd-relax](https://github.com/src-d/wmd-relax) - Python package which finds nearest neighbors at Word Mover's Distance.


<a name="datasets"></a>
#### Datasets
* [GitHub repositories - languages distribution](https://data.world/source-d/github-repositories-languages-distribution) - Programming languages distribution in 14,000,000 repositories on GitHub (October 2016).
* [452M commits on GitHub](https://data.world/vmarkovtsev/452-m-commits-on-github) - ≈ 452M commits' metadata from 16M repositories on GitHub (October 2016).
* [GitHub readme files](https://data.world/vmarkovtsev/github-readme-files) - Readme files of all GitHub repositories (16M) (October 2016).
* [from language X to Y](https://data.world/vmarkovtsev/from-language-x-to-y) - Cache file Erik Bernhardsson collected for his awesome blog post.
* [GitHub word2vec 120k](https://data.world/vmarkovtsev/github-word-2-vec-120-k) - Sequences of identifiers extracted from top starred 120,000 GitHub repos.
* [GitHub Source Code Names](https://data.world/vmarkovtsev/github-source-code-names) - Names in source code extracted from 13M GitHub repositories, not people.
* [GitHub duplicate repositories](https://data.world/vmarkovtsev/github-duplicate-repositories) - GitHub repositories not marked as forks but very similar to each other.
* [GitHub lng keyword frequencies](https://data.world/vmarkovtsev/github-lng-keyword-frequencies) - Programming language keyword frequency extracted from 16M GitHub repositories.
* [GitHub Java Corpus](http://groups.inf.ed.ac.uk/cup/javaGithub/ ) - GitHub Java corpus is a set of Java projects collected from GitHub that we have used in a number of our publications. The corpus consists of 14,785 projects and 352,312,696 LOC.
* [150k Python Dataset](http://www.srl.inf.ethz.ch/py150.php) - Dataset consisting of 150'000 Python ASTs.
* [150k JavaScript Dataset](http://www.srl.inf.ethz.ch/js150.php) - Dataset consisting of 150'000 JavaScript files and their parsed ASTs.
* [card2code](https://github.com/deepmind/card2code) - This dataset contains the language to code datasets described in the paper [Latent Predictor Networks for Code Generation](https://arxiv.org/abs/1603.06744).

<a name="credits"></a>
## Credits

* A lot of references and articles were taken from [mast-group](https://mast-group.github.io/)

<a name="contributions"></a>
## Contributions

See [CONTRIBUTING.md](CONTRIBUTING.md).

<a name="license"></a>
## License

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
