import torch
import subprocess

class GPUSelector:
    
    def __init__(self):
        self.num_gpus = torch.cuda.device_count()
        self.gpu_info = self.get_gpu_info()

    
    def format_gpu_info(self, result) -> list[tuple]:
        info_dict = {}
        split_gpu_info = result.split('\n')[1:-1]
        for gpu_index, gpu_info in enumerate(split_gpu_info):
            total, used, free = gpu_info.split(', ')
            total = int(total.split(' ')[0])
            used = int(used.split(' ')[0])
            free = int(free.split(' ')[0])
            info_dict[gpu_index] = (total, used, free)
        return info_dict
            
            
    def get_gpu_info(self):
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv"],
                                capture_output=True,
                                text=True).stdout
        formatted_result = self.format_gpu_info(result)
        return formatted_result
        
        
    def select_gpu(self):
        print('GPU info:')
        for gpu_index, (total, used, free) in self.gpu_info.items():    
            print(f'GPU {gpu_index}: Total memory: {total} MiB, Used memory: {used} MiB, Free memory: {free} MiB')
                
        selected_gpu = -1
        while True:
            try:
                selected_gpu = int(input(f'Select a GPU (0-{self.num_gpus-1}): '))
                if selected_gpu not in range(self.num_gpus):
                    raise ValueError
                break
            except ValueError:
                print(f'Invalid input. Please enter a number between 0 and {self.num_gpus-1}')
                
        return selected_gpu
            
    
        
    
        
