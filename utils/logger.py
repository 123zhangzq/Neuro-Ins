import torch
import math
    
def log_to_screen(time_used, count_obj_ci, average_diff_obj_ci, count_obj_mm, average_diff_obj_m,
                  batch_size, dataset_size):
    # reward
    print('\n', '-'*60)
    print('Number of instances no worse than cheapest insertion:'.center(35), '{:f}s'.format(count_obj_ci))
    print('Avg difference:'.center(35),
          '{:f}s'.format(average_diff_obj_ci))
    print('Number of instances no worse than MP model:'.center(35),
          '{:f}s'.format(count_obj_mm))
    print('Avg difference:'.center(35),
          '{:f}s'.format(average_diff_obj_m))


    
    # time
    print('-'*60)
    print('Avg used time:'.center(35), '{:f}s'.format(
            time_used.mean() / dataset_size))
    print('-'*60, '\n')


def log_to_screen_and_file(time_used, count_obj_ci, average_diff_obj_ci, count_obj_mm, average_diff_obj_m,
                           batch_size, dataset_size, output_file_path, epoch):
    # reward
    with open(output_file_path, 'a') as file:
        file.write('\n' + '-'*60 + '\n')
        file.write(str(epoch))
        file.write('\n' + '-' * 60 + '\n')
        file.write('The number of instances not worse than cheapest insertion:'.center(35) + '{:f}\n'.format(count_obj_ci))
        file.write('Avg difference:'.center(35) + '{:f}\n'.format(average_diff_obj_ci))
        file.write('The number of instances not worse than math model:'.center(35) + '{:f}\n'.format(count_obj_mm))
        file.write('Avg difference:'.center(35) + '{:f}\n'.format(average_diff_obj_m))

        # time
        file.write('-'*60 + '\n')
        file.write('Avg used time:'.center(35) + '{:f}s\n'.format(time_used.mean() / dataset_size))
        file.write('-'*60 + '\n\n')

    # Output to screen
    print('\n', '-'*60)
    print('The number of instances not worse than cheapest insertion:'.center(35), '{:f}'.format(count_obj_ci))
    print('Avg difference:'.center(35), '{:f}'.format(average_diff_obj_ci))
    print('The number of instances not worse than math model:'.center(35), '{:f}'.format(count_obj_mm))
    print('Avg difference:'.center(35), '{:f}'.format(average_diff_obj_m))
    print('-'*60)
    print('Avg used time:'.center(35), '{:f}'.format(time_used.mean() / dataset_size))
    print('-'*60, '\n')


    
def log_to_tb_val(tb_logger, time_used, init_value, best_value, reward, costs_history, search_history,
                  batch_size, val_size, dataset_size, T, epoch):
        
    tb_logger.log_value('validation/avg_time',  time_used.mean() / dataset_size, epoch)
    tb_logger.log_value('validation/avg_total_reward', reward.sum(1).mean(), epoch)
    tb_logger.log_value('validation/avg_step_reward', reward.mean(), epoch)
    
    
    tb_logger.log_value(f'validation/avg_init_cost', init_value.mean(), epoch)
    tb_logger.log_value(f'validation/avg_best_cost', best_value.mean(), epoch)

    for per in range(20,100,20):
        cost_ = costs_history[:,round(T*per/100)]
        tb_logger.log_value(f'validation/avg_.{per}_cost', cost_.mean(), epoch)

def log_to_tb_train(tb_logger, agent, Reward, ratios, bl_val_detached, total_cost, grad_norms, reward, entropy, approx_kl_divergence,
               reinforce_loss, baseline_loss, log_likelihood, initial_cost, mini_step):
    
    tb_logger.log_value('learnrate_pg', agent.optimizer.param_groups[0]['lr'], mini_step)            
    avg_cost = (total_cost).mean().item()
    tb_logger.log_value('train/avg_cost', avg_cost, mini_step)
    tb_logger.log_value('train/Target_Returen', Reward.mean().item(), mini_step)
    tb_logger.log_value('train/ratios', ratios.mean().item(), mini_step)
    avg_reward = torch.stack(reward, 0).sum(0).mean().item()
    max_reward = torch.stack(reward, 0).max(0)[0].mean().item()
    tb_logger.log_value('train/avg_reward', avg_reward, mini_step)
    tb_logger.log_value('train/init_cost', initial_cost.mean(), mini_step)
    tb_logger.log_value('train/max_reward', max_reward, mini_step)
    grad_norms, grad_norms_clipped = grad_norms
    tb_logger.log_value('loss/actor_loss', reinforce_loss.item(), mini_step)
    tb_logger.log_value('loss/nll', -log_likelihood.mean().item(), mini_step)
    tb_logger.log_value('train/entropy', entropy.mean().item(), mini_step)
    tb_logger.log_value('train/approx_kl_divergence', approx_kl_divergence.item(), mini_step)
    tb_logger.log_histogram('train/bl_val',bl_val_detached.cpu(),mini_step)
    
    tb_logger.log_value('grad/actor', grad_norms[0], mini_step)
    tb_logger.log_value('grad_clipped/actor', grad_norms_clipped[0], mini_step)
    tb_logger.log_value('loss/critic_loss', baseline_loss.item(), mini_step)
            
    tb_logger.log_value('loss/total_loss', (reinforce_loss+baseline_loss).item(), mini_step)
    
    tb_logger.log_value('grad/critic', grad_norms[1], mini_step)
    tb_logger.log_value('grad_clipped/critic', grad_norms_clipped[1], mini_step)