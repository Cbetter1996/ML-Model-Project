# ML-Model-Project  
本文档用于收录在学习机器学习和深度学习过程中的一些项目经验，仅供参考，如有错误请留言&指正。  

## Part 1-Environment  
### 整体环境配置  
Mac版本：MacOS Big Sur 11.4  
Python版本：3.8.5  
Tensorflow版本：2.4.1  
Pip版本：23.2.1  
Numpy版本：1.19.5  
Pandas版本：1.4.3  

### Tensorflow安装  
经过好几天的失败尝试，终于把tensorflow安装成功和跑通。   
Step 1: python --version检查python版本，如果非3.8.5可以直接安装conda install python=3.8.5。     
Step 2: pip install virtualenv，安装虚拟环境的包，如有可以直接跳过这一步。    
Step 3: virtualenv tensorflow，创建一个名称为tensorflow的虚拟环境。    
Step 4: source activate tensorflow，激活虚拟环境。  
Step 5: 从这个地址下载tensorflow包，存在本地路径上。  
Step 6: pip install ~/Downloads/tensorflow-2.4.1-py3-none-any.whl，安装本地路径的包。  
Step 7: 可选，安装完tensorflow后发现numpy和pandas版本冲突无法使用，如果发现冲突可以针对pandas的版本进行调整。  
其他相关安装流程，[可参考](https://pianshen.com/ask/530814350740/)  


## Part 2-Model Learning Project
### Text Classification Based on Machine Learning  
相关项目，[可参照](https://pianshen.com/ask/530814350740/](https://github.com/Alic-yuan/nlp-beginner-finish)https://github.com/Alic-yuan/nlp-beginner-finish)  




