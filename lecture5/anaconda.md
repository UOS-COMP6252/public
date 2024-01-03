To list all environments
```conda env list ```
To remove(delete) an environment called E
``` conda env remove -n E ```
To replicate an environment called E (for example to tell students to ???)
```
conda activate E   
conda env export > env.yml

```

To use the above ```env.yml``` to create another environment G

```conda env create -n G -f env.yml```