import re
import pandas as pd
import warnings

import time
import os

os.environ['NCCL_P2P_DISABLE'] = '1'
#os.environ["WORLD_SIZE"] = "8"
os.environ["NCCL_DEBUG"] = "WARN"

warnings.simplefilter(action='ignore', category=FutureWarning)

class MonteCarloTreeSearch: 
    def __init__(self, model,tokenizer, sampling_params, search_limit=12, alpha=0.5, beta=0.9, L=500, cpuct=0.125):
        self.tokenizer = tokenizer
        self.model = model
        self.sampling_params = sampling_params
        self.pool = [] 
        self.search_limit = search_limit
        self.alpha = alpha
        self.beta = beta
        self.L = L
        self.cpuct = cpuct
        self.tree = []

        self.messages = [
            {'role':'system','content':"You're an obediant mathematical assistant, You will answer math questions and start a new line after each step. Your final answer must be encapsulated with \boxed{final answer}. You must rely on preivous answers to format the final result you will give in \boxed{final answer}."},
            {'role':'user','content':"""What is the range of the function $y = \frac{x^2 + 3x + 2}{x+1}$? (Express your answer using interval notation.)"""},
            {'role':'assistant','content':"""We can factor the numerator to get $y = \frac{(x+1)(x+2)}{x+1}$. \n If we exclude the case where $x = -1$, the function is equivalent to $y = x+2$. \n However, because $x$ cannot equal $-1$, $y$ cannot equal 1. \n Therefore, the range is all real numbers except for 1, which we may write as $y \in \boxed{(-\infty, 1)\cup(1, \infty)}.$"""},
            {'role':'user','content':"""Let \[f(x) = \left\{ \begin{array}{cl} ax+3, &\text{ if }x>2, \\ x-5 &\text{ if } -2 \le x \le 2, \\ 2x-b &\text{ if } x <-2. \end{array} \right.\]Find $a+b$ if the piecewise function is continuous (which means that its graph can be drawn without lifting your pencil from the paper)."""},
            {'role':'assistant','content':"""For the piecewise function to be continuous, the cases must "meet" at $2$ and $-2$. \n For example, $ax+3$ and $x-5$ must be equal when $x=2$. \n This implies $a(2)+3=2-5$, which we solve to get $2a=-6 \Rightarrow a=-3$. \n Similarly, $x-5$ and $2x-b$ must be equal when $x=-2$. \n Substituting, we get $-2-5=2(-2)-b$, which implies $b=3$. So $a+b=-3+3=\boxed{0}$."""},
            {'role':'user','content':"""Suppose $a$, $b,$ and $c$ are positive numbers satisfying: \begin{align*} a^2/b &= 1, \\ b^2/c &= 2, \text{ and}\\ c^2/a &= 3. \end{align*}"""},
            {'role':'assistant','content':"""Notice that multiplying all three of the original equations together tells us that $(a^2b^2c^2)/(abc) = 6$, which implies $abc=6$. \n Rewriting the first and third equations as $b = a^2$ and $c = \sqrt{3a}$ and plugging these into $abc=6$ yields $a \cdot a^2\cdot \sqrt{3a} = 6$. \n By squaring both sides of the equation, we obtain $3a^7 = 36 \Rightarrow a = \boxed{12^{1/7}}$."""},
            {'role':'user','content':"""An infinite geometric series has a first term of $12$ and a second term of $4.$ A second infinite geometric series has the same first term of $12,$ a second term of $4+n,$ and a sum of four times that of the first series. Find the value of $n.$"""},
            {'role':'assistant','content':"""Note that if the the two series have constant ratios of $a$ and $b,$ respectively, then $4\left( \frac{12}{1-a} \right) = \frac{12}{1-b}.$ \n Simplifying, $4(1-b)=1-a.$ Substituting in $a= \frac{4}{12}=\frac{1}{3}$ and $b= \frac{4+n}{12}=\frac{1}{3}+\frac{n}{12},$ we quickly find that $n=\boxed{6}.$"""},
        ]

        
    def create_tree(self,dataset,repetion,nmbr):
        start = time.time()
        columns = ['question', 'state','rollouts','nbr_visits','mc_estimation']
        tree_df = pd.DataFrame(columns=columns)
        for i in range(1600,1600+nmbr):
            
            count_tree=0
            
            golden_answer = re.findall(r'boxed{([^}]*)}', dataset['ground_truth_answer'][i].replace('\boxed','boxed'))[0]
            question, golden_answer, seed, correctness = dataset['question'][i].strip(), golden_answer, dataset['answer'][i], dataset['correctness'][i]
            self.pool = []
            self.tree = []

            tree = self.omega_prm(question, golden_answer, seed, correctness) #can be empty if no solution is found

            if tree == None:
                continue

            if self.pool != [] and self.tree != []:
                while count_tree < repetion or not self.pool:  # Constructing the tree
                    count_tree+=1

                    state, rollout = self.selection()
                    if state == False:
                        break
                    
                    solution = state + rollout

                    self.binary_search(question, solution, golden_answer)

                    for state in self.tree:
                        question, prefix, rollouts, visits, mc = state[0], state[1], state[2], state[3], state[4]
                       


                        new_row = pd.DataFrame({'question': [question], 'state': [prefix], 'rollouts': [rollouts], 'nbr_visits': [visits], 'mc_estimation': [mc]})
                        tree_df = pd.concat([tree_df, new_row], ignore_index=True)

        end =time.time()
        tree_time = end - start
        return tree_df, tree_time
        
    
    def selection(self):
        
        max_value = 0
        usable=False
        for i in range(len(self.pool)):
            
            state,rollout,value = self.pool.pop()
            player = self.ucb1(state,rollout)

            if player>=max_value :
                if i > 0:
                    self.pool = [(max_state,max_rollout,max_value)] + self.pool  

                max_state,max_rollout,max_value=state,rollout,value
                max_value=player
                usable=True
                
            else:
                self.pool =[(state,rollout,value)] + self.pool
                
        if usable:
            for i in range(len(self.tree)):
                if max_state == self.tree[i][1]: 
                    self.tree[i][3]=int(self.tree[i][3])+1 

            return max_state,max_rollout #
        else:
            return False, False

    def ucb1(self, state,rollout): 
        total_visit = 0
        for i in range(len(self.tree)):
            
            total_visit+= int(self.tree[i][3])
            if state == self.tree[i][1]: 
                N,MC = int(self.tree[i][3]),float(self.tree[i][4])
                #self.tree[i][3]=int(self.tree[i][3])+1 #depends


        return (self.cpuct * (total_visit ** 0.5) / (1+N))+self.alpha**(1-MC)*self.beta**(len(rollout)/self.L)
    

    def monte_carlo_estimation(self, question, solution_prefix,golden_answer):
        """
        Perform Monte Carlo rollouts from the current solution prefix.
        """
        mc=0
        NU=0
        rollouts = []
        
        solution_prefix = " \n ".join(solution_prefix)
        
        list_rollout_completion = self.create_rollout_complete_solution(question,solution_prefix)
        if list_rollout_completion ==0:
            return 0
        for rollout in list_rollout_completion:
            
            correct = self.compare_with_golden_answer(rollout, golden_answer)
            
            if correct == "not usable":
                NU+=1
                continue
            else:       
                rollouts.append((solution_prefix, rollout, correct))
                if correct:
                    mc +=1
        
        mc_estimation = mc/(8*self.search_limit-NU+1e-7)

        if mc_estimation>0 and mc_estimation<1: #
   
            self.pool += rollouts
            self.complete_tree(question,solution_prefix,rollouts,mc_estimation)
        
        return mc_estimation
    
 
    
    def create_rollout_complete_solution(self,question,solution_prefix):
        count=0
        
        prompts_all = [self.messages+[{'role':'user','content':question},{'role':'assistant','content':solution_prefix}] for _ in range(8)]*self.search_limit
        prompts_all = [self.tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True) for prompt in prompts_all]

        results_gathered = self.accelerate_construction(prompts_all)
        return results_gathered

    
    def compare_with_golden_answer(self, rollout,golden_answer):
        """
        Compare the generated rollout with the golden (correct) answer.
        """

        answer = rollout.replace('\boxed','boxed')
        try:
            answer = re.findall(r'boxed{([^}]*)}', answer)[0]
            try:
                return float(answer) == float(golden_answer)
            except:
                return float(answer) == float(golden_answer)
                    
                
        except:
            
            return "not usable"
        
        

    def split_response(self,string):
        list_sentences = string.split('\n')
        return [sentence for sentence in list_sentences if len(sentence)>10] 
    
    def binary_search(self, question, solution,golden_answer):
        """
        Use binary search to find the first incorrect step in the solution.
        """
        solution = self.split_response(solution)
        left, right = 0, len(solution)
        
        while  left < right : 
            mid = (left + right) // 2
            mc_estimation = self.monte_carlo_estimation(question, solution[:mid],golden_answer) #All divide-and-rollout positions before the first error become new states.
            if mc_estimation>0:
                left = mid + 1
            else:
                right = mid



    def complete_tree(self,question,state,rollouts,MC):

        already_there = False

        for i in range(len(self.tree)):
            if state == self.tree[i][1]:
                self.tree[i][4]=MC
                self.tree[i][3]=int(self.tree[i][3])+1 
                already_there=True
                for rollout in rollouts:
                    if rollout not in self.tree[i][2]:
                        
                        self.tree[i][2]+=[rollout] 

        
        if already_there==False:
            self.tree+= [[question,state,[rollouts[0]],1,MC]] 
            for rollout in rollouts:
                    if rollout not in self.tree[-1][2]:
                        
                        self.tree[-1][2]+=[rollout]
            


    def omega_prm(self, question,golden_answer,seed,correctness):
        """
        Core function to run the OmegaPRM algorithm and build the state-action tree.
        """

        if not correctness: #only use wrong answers as described in OmegaPRM paper but can correct answers can also be used.

            self.binary_search(question, seed, golden_answer) 
        else:
            return None

        return self.tree
        


    def accelerate_construction(self, prompts_all):
        """
        Generate text in parallel using multiple GPUs.
        """

        generated_text=[]
            
        outputs = self.model.generate(prompts_all, self.sampling_params)
        for output in outputs:
            generated_text.append(output.outputs[0].text)
        
        return generated_text
            

    
    