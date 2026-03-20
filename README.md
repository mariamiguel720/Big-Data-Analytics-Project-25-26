# Big-Data-Analytics-Project-25-26

## Como trabalhar com o git passo a passo:

### PASSO 1: 
Antes de começares a trabalhar vê em que branch estás: 
 
git branch
 
 
Se já estiveres no teu branch, avança para o passo 2. 
Se não estiveres, muda para o teu branch com: 
 
 
git checkout nome-do-teu-branch
 
 
### PASSO 2:

git pull origin nome-branch-comum 
 
Agora estás pronto para trabalhar à vontade
 
 
### PASSO 3: 
Quando acabares o trabalho 

git add . 
git commit -m "mensagem explicativa do commit" 
git push origin nome-do-teu-branch
 
 
### PASSO 4:
Atualizar o branch comum 

git checkout nome-branch-comum  
git pull origin nome-branch-comum  
git merge nome-do-teu-branch
git push nome-branch-comum
