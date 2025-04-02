import numpy as np
import time
import timeit
from operator import itemgetter
import os
import ray
import thinker.util as util
from thinker.core.file_writer import FileWriter
import torch
AB_CAN_WRITE, AB_FULL, AB_FINISH = 0, 1, 2

def custom_choice(tran_n, batch_size, p, replace=False):
    if np.any(np.isnan(p)):
        p[np.isnan(p)] = 0.01
        p /= p.sum()

    non_zero_count = np.count_nonzero(p)
    if non_zero_count < batch_size and not replace:
        # Set zero probabilities to 0.01
        zero_indices = np.where(p == 0)[0]
        p[zero_indices] = 0.01
        # Scale the remaining probabilities
        p /= p.sum()

    return np.random.choice(tran_n, batch_size, p=p, replace=replace)


@ray.remote
class ActorBuffer:
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.buffer = []
        self.buffer_state = []
        self.finish = False

    def write(self, data, state):
        # Write data, a named tuple of numpy arrays each with shape (t, b, ...)
        # and state, a tuple of numpy arrays each with shape (*,b, ...)
        self.buffer.append(data)
        self.buffer_state.append(state)

    def available_for_read(self):
        # Return True if the total size of data in the buffer is larger than the batch size
        total_size = sum([data[0].shape[1] for data in self.buffer])
        return total_size >= self.batch_size

    def get_status(self):
        # Return AB_FULL if the number of data inside the buffer is larger than 3 * self.batch_size
        if self.finish:
            return AB_FINISH
        total_size = sum([data[0].shape[1] for data in self.buffer])
        if total_size >= 2 * self.batch_size:
            return AB_FULL
        else:
            return AB_CAN_WRITE

    def read(self):
        # Return a named tuple of numpy arrays, each with shape (t, batch_size, ...)
        # from the first n=batch_size data in the buffer and remove it from the buffer

        if not self.available_for_read():
            return None

        collected_data = []
        collected_state = []
        collected_size = 0
        n_tuple = type(self.buffer[0])
        while collected_size < self.batch_size and self.buffer:
            collected_data.append(self.buffer.pop(0))
            collected_state.append(self.buffer_state.pop(0))
            collected_size += collected_data[-1][0].shape[1]

        # Concatenate the named tuple of numpy arrays along the batch dimension
        output = n_tuple(
            *(
                np.concatenate([data[i] for data in collected_data], axis=1)
                if collected_data[0][i] is not None
                else None
                for i in range(len(collected_data[0]))
            )
        )
        output_state = tuple(
            np.concatenate([state[i] for state in collected_state], axis=0)
            for i in range(len(collected_state[0]))
        )

        # If the collected data size is larger than the batch size, store the extra data back to the buffer
        if collected_size > self.batch_size:
            extra_data = n_tuple(
                *(
                    data[:, -collected_size + self.batch_size :, ...]
                    if data is not None
                    else None
                    for data in output
                )
            )
            self.buffer.insert(0, extra_data)
            extra_state = tuple(
                state[-collected_size + self.batch_size :, ...]
                for state in output_state
            )
            self.buffer_state.insert(0, extra_state)

            # Trim the output to have the exact batch size
            output = n_tuple(
                *(
                    data[:, : self.batch_size, ...] if data is not None else None
                    for data in output
                )
            )
            output_state = tuple(
                state[: self.batch_size, ...] for state in output_state
            )

        return output, output_state

    def set_finish(self):
        self.finish = True


class SModelBuffer:
    def __init__(self, flags):
        self.alpha = flags.priority_alpha
        self.t = flags.buffer_traj_len
        self.k = flags.model_unroll_len
        self.ret_n = flags.model_return_n
        self.max_buffer_n = flags.model_buffer_n // self.t + 1  # maximum buffer length
        self.batch_size = flags.model_batch_size  # batch size in returned sample
        self.wram_up_n = (
            flags.model_warm_up_n if not flags.ckp else flags.model_buffer_n
        )  # number of total transition before returning samples

        self.frame_stack_n = 1
        self.pf = 0 # frame_stack_n - 1
        self.mrl = flags.model_mem_unroll_len

        self.buffer = []
        
        self.abs_processed_traj = 0 # number of absolute processed trajectory so far
        self.rel_processed_traj = 0 # number of relative processed trajectory so far
        self.priorities = None # has shape (rel_processed_traj * t,)
        self.buffer_ind = None 
        # has shape (rel_processed_traj, rel_processed_traj) which corresponds
        # to the two indexes of the processed trajectory in self.buffer

        self.base_ind = 0
        self.clean_m = 0
        self.finish = False

    def write(self, data):
        # data is a named tuple with elements of size (pf+t+k+ret_n+1, b, ...)
        b = data[0].shape[1]

        p_shape = self.t * b
        if self.priorities is None:
            self.priorities = np.ones((p_shape), dtype=float)
        else:
            max_priorities = np.full(
                (p_shape), fill_value=self.priorities.max(), dtype=float
            )
            self.priorities = np.concatenate([self.priorities, max_priorities])

        new_buffer_ind = np.hstack((np.full((b, 1), len(self.buffer)), np.arange(b)[:, None]))
        if self.buffer_ind is None:
            self.buffer_ind = new_buffer_ind
        else:
            self.buffer_ind = np.concatenate([self.buffer_ind, new_buffer_ind])
        
        self.buffer.append(data)
        self.abs_processed_traj += b
        self.rel_processed_traj += b

        # clean periordically
        self.clean()

    def read(self, beta):
        if self.priorities is None or self.abs_processed_traj * self.t < self.wram_up_n:
            return None
        if self.finish:
            return "FINISH"
        return self.prepare(beta)

    def prepare(self, beta):
        tran_n = len(self.priorities) # should be rel_processed_traj * t
        probs = self.priorities**self.alpha
        probs /= max(probs.sum(), 1e-8)
        flat_ind = custom_choice(tran_n, self.batch_size, p=probs, replace=False)
        ind = np.unravel_index(flat_ind, (self.rel_processed_traj, self.t))
        # we need to futher unravel the inds[0] manually
        b_ind = self.buffer_ind[ind[0]]
        ind_0, ind_1 = b_ind[:, 0], b_ind[:, 1]
        ind_2 = ind[1]

        weights = (tran_n * probs[flat_ind]) ** (-beta)
        weights /= weights.max()

        data = {}
        for d, field in enumerate(self.buffer[0]._fields):
            if field == "initial_per_state": continue
            elems = []
            st_pd = 0 if field == "real_state" else self.pf # we need also get the previous frame_stack_n - 1 states
            for i in range(self.batch_size):
                sel_elem = self.buffer[ind_0[i]][d]
                elems.append(
                    sel_elem[self.mrl + st_pd + ind_2[i] : self.mrl + self.pf + ind_2[i] + self.k + self.ret_n + 2, [ind_1[i]]]
                )
            data[field] = np.concatenate(elems, axis=1)

        if self.pf > 0:
            done = np.logical_or(data["done"], data["truncated_done"])
            data["real_state"] = stack_frame(data["real_state"], self.pf + 1, done)            
        
        initial_per_state = {}
        for field in self.buffer[0].initial_per_state.keys():
            elems = []
            for i in range(self.batch_size):
                sel_elem = self.buffer[ind_0[i]].initial_per_state[field]
                elems.append(
                    sel_elem[self.pf + ind_2[i] : self.pf + ind_2[i] + 1, [ind_1[i]]]
                )
            initial_per_state[field] = np.concatenate(elems, axis=1)

        if self.mrl > 0:
            past_real_state = []
            past_done = []
            past_action = []
            for i in range(self.batch_size):
                sel_elem = self.buffer[ind_0[i]].real_state
                past_real_state.append(
                    sel_elem[ind_2[i] : self.mrl + self.pf + ind_2[i], [ind_1[i]]]
                )
                sel_elem = self.buffer[ind_0[i]].done
                past_done.append(
                    sel_elem[self.pf + ind_2[i] : self.mrl + self.pf + ind_2[i], [ind_1[i]]]
                )
                sel_elem = self.buffer[ind_0[i]].action
                past_action.append(
                    sel_elem[self.pf + ind_2[i] : self.mrl + self.pf + ind_2[i], [ind_1[i]]]
                )
            past_real_state = np.concatenate(past_real_state, axis=1)            
            if self.pf > 0:
                past_real_state = stack_frame(past_real_state, self.pf + 1, done)   
            past_done = np.concatenate(past_done, axis=1)
            past_action = np.concatenate(past_action, axis=1)
            initial_per_state["past_real_state"] = past_real_state
            initial_per_state["past_done"] = past_done
            initial_per_state["past_action"] = past_action
        
        data["initial_per_state"] = initial_per_state        
        data = type(self.buffer[0])(**data)
        base_ind_pri = self.base_ind * self.t
        abs_flat_inds = flat_ind + base_ind_pri
        return data, weights, abs_flat_inds, self.abs_processed_traj * self.t

    def set_finish(self):
        self.finish = True

    def get_status(self):
        return {"processed_n": self.abs_processed_traj * self.t,
                "warm_up_n": self.wram_up_n,
                "running": self.abs_processed_traj * self.t >= self.wram_up_n,
                "finish": self.finish,
                 }

    def set_frame_stack_n(self, frame_stack_n):
        self.frame_stack_n = frame_stack_n
        self.pf = frame_stack_n - 1 

    def update_priority(self, abs_flat_inds, priorities):
        """Update priority in the buffer; both input
        are np array of shape (model_batch_size,)"""
        base_ind_pri = self.base_ind * self.t
        flat_inds = abs_flat_inds - base_ind_pri  # get the relative index        
        mask = flat_inds >= 0        
        flat_inds = flat_inds[mask]
        priorities = priorities[mask]
        self.priorities[flat_inds] = priorities

    def clean(self):        
        if self.rel_processed_traj > self.max_buffer_n:
            excess_n = self.buffer_ind[self.rel_processed_traj - self.max_buffer_n, 0] + 1
            excess_m = np.sum(self.buffer_ind[:, 0] <= excess_n - 1)
            del self.buffer[:excess_n]
            self.buffer_ind = self.buffer_ind[excess_m:]
            self.priorities = self.priorities[excess_m * self.t :]            
            self.buffer_ind[:, 0] = self.buffer_ind[:, 0] - excess_n
            self.rel_processed_traj -= excess_m
            self.base_ind += excess_m

def stack_frame(frame, frame_stack_n, done):
    T, B, C, H, W = frame.shape[0] - frame_stack_n + 1, frame.shape[1], frame.shape[2], frame.shape[3], frame.shape[4]
    assert done.shape[0] == T
    y = np.zeros((T, B, C * frame_stack_n, H, W), dtype=frame.dtype)
    for s in range(frame_stack_n):
        y[:, :, s*C:(s+1)*C, :, :] = frame[s:T+s]
        y[:, :, s*C:(s+1)*C, :, :][done] = frame[frame_stack_n - 1: T + frame_stack_n - 1][done]
    return y

@ray.remote
class GeneralBuffer(object):
    def __init__(self):
        self.data = {}

    def extend_data(self, name, x):
        if name in self.data:
            self.data[name].extend(x)
        else:
            self.data[name] = x
        return self.data[name]

    def update_dict_item(self, name, key, value):
        if not name in self.data:
            self.data[name] = {}
        self.data[name][key] = value
        return True

    def set_data(self, name, x):
        self.data[name] = x
        return True

    def get_data(self, name):
        return self.data[name] if name in self.data else None
    
    def get_and_increment(self, name):
        if name in self.data:
            self.data[name] += 1            
        else:
            self.data[name] = 0
        return self.data[name]

@ray.remote
class ModelBuffer(SModelBuffer):
    pass

@ray.remote
class RecordBuffer(object):
    # simply for logging return from self play thread if actor learner is not running
    def __init__(self, flags):
        self.flags = flags
        self.flags.git_revision = util.get_git_revision_hash()
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )
        self.real_step = 0
        max_actor_id = (
            self.flags.gpu_num_actors * self.flags.gpu_num_p_actors
            + self.flags.cpu_num_actors * self.flags.cpu_num_p_actors
        )
        self.last_returns = RetBuffer(max_actor_id, mean_n=400)
        self._logger = util.logger()
        self.preload = flags.ckp
        self.preload_n = flags.model_buffer_n
        self.proc_n = 0

    def set_real_step(self, x):
        self.real_step = x

    def insert(self, episode_return, episode_step, real_done, actor_id):
        T, B, *_ = episode_return.shape
        self.proc_n += T*B
        if self.preload and self.proc_n < self.preload_n:
            self._logger.info("[%s] Preloading: %d / %d" % (self.flags.xpid, self.proc_n, self.preload_n,))
            return self.real_step
        
        self.real_step += T*B        
        if np.any(real_done):
            episode_returns = episode_return[real_done]
            episode_returns = tuple(episode_returns)
            episode_lens = episode_step[real_done]
            episode_lens = tuple(episode_lens)
            done_ids = np.broadcast_to(actor_id, real_done.shape)[
                real_done
            ]
            done_ids = tuple(done_ids)
        else:
            episode_returns, episode_lens, done_ids = (), (), ()
        self.last_returns.insert(episode_returns, done_ids)
        rmean_episode_return = self.last_returns.get_mean()
        stats = {
            "real_step": self.real_step,
            "rmean_episode_return": rmean_episode_return,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
        }
        self.plogger.log(stats)
        print_str = ("[%s] Steps %i @ Ret %f." % (self.flags.xpid, self.real_step, stats["rmean_episode_return"],))
        self._logger.info(print_str)
        return self.real_step

class RetBuffer:
    def __init__(self, max_actor_id, mean_n=400):
        """
        Compute the trailing mean return by storing the last return for each actor
        and average them;
        Args:
            max_actor_id (int): maximum actor id
            mean_n (int): size of data for computing mean
        """
        buffer_n = mean_n // max_actor_id + 1
        self.return_buffer = np.zeros((max_actor_id, buffer_n))
        self.return_buffer_pointer = np.zeros(
            max_actor_id, dtype=int
        )  # store the current pointer
        self.return_buffer_n = np.zeros(
            max_actor_id, dtype=int
        )  # store the processed data size
        self.max_actor_id = max_actor_id
        self.mean_n = mean_n
        self.all_filled = False

    def insert(self, returns, actor_ids):
        """
        Insert new returnss to the return buffer
        Args:
            returns (tuple): tuple of float, the return of each ended episode
            actor_ids (tuple): tuple of int, the actor id of the corresponding ended episode
        """
        # actor_id is a tuple of integer, corresponding to the returns
        if len(returns) == 0:
            return
        assert len(returns) == len(actor_ids)
        for r, actor_id in zip(returns, actor_ids):
            if actor_id >= self.max_actor_id:
                continue
            # Find the current pointer for the actor
            pointer = self.return_buffer_pointer[actor_id]
            # Update the return buffer for the actor with the new return
            self.return_buffer[actor_id, pointer] = r
            # Update the pointer for the actor
            self.return_buffer_pointer[actor_id] = (
                pointer + 1
            ) % self.return_buffer.shape[1]
            # Update the processed data size for the actor
            self.return_buffer_n[actor_id] = min(
                self.return_buffer_n[actor_id] + 1, self.return_buffer.shape[1]
            )

        if not self.all_filled:
            # check if all filled
            self.all_filled = np.all(
                self.return_buffer_n >= self.return_buffer.shape[1]
            )

    def insert_raw(self, episode_returns, ind, actor_id, done):                
        if torch.is_tensor(episode_returns):
            episode_returns = episode_returns.detach().cpu().numpy()
        if torch.is_tensor(actor_id):
            actor_id = actor_id.detach().cpu().numpy()
        if torch.is_tensor(done):
            done = done.detach().cpu().numpy()

        episode_returns = episode_returns[done][:, ind]
        episode_returns = tuple(episode_returns)
        done_ids = np.broadcast_to(actor_id, done.shape)[done]
        done_ids = tuple(done_ids)
        self.insert(episode_returns, done_ids)

    def get_mean(self):
        """
        Compute the mean of the returns in the buffer;
        """
        if self.all_filled:
            overall_mean = np.mean(self.return_buffer)
        else:
            # Create a mask of filled items in the return buffer
            col_indices = np.arange(self.return_buffer.shape[1])
            # Create a mask of filled items in the return buffer
            filled_mask = (
                col_indices[np.newaxis, :] < self.return_buffer_n[:, np.newaxis]
            )
            if np.any(filled_mask):
                # Compute the sum of returns for each actor considering only filled items
                sum_returns = np.sum(self.return_buffer * filled_mask)
                # Compute the mean for each actor by dividing the sum by the processed data size
                overall_mean = sum_returns / np.sum(filled_mask.astype(float))
            else:
                overall_mean = 0.0
        return overall_mean

@ray.remote
class SelfPlayBuffer:    
    def __init__(self, flags):
        # A ray actor tailored for logging across self-play worker; code mostly from learn_actor
        self.flags = flags
        self._logger = util.logger()

        max_actor_id = (
            self.flags.self_play_n * self.flags.env_n
        )

        self.ret_buffers = [RetBuffer(max_actor_id, mean_n=400)]
        if self.flags.im_cost > 0.:
            self.ret_buffers.append(RetBuffer(max_actor_id, mean_n=20000))
        if self.flags.cur_cost > 0.:
            self.ret_buffers.append(RetBuffer(max_actor_id, mean_n=400))      
        
        self.plogger = FileWriter(
            xpid=flags.xpid,
            xp_args=flags.__dict__,
            rootdir=flags.savedir,
            overwrite=not self.flags.ckp,
        )

        self.rewards_ls = ["re"]
        if flags.im_cost > 0.0:
            self.rewards_ls += ["im"]
        if flags.cur_cost > 0.0:
            self.rewards_ls += ["cur"]
        self.num_rewards = len(self.rewards_ls)

        self.step, self.real_step, self.tot_eps = 0, 0, 0        
        self.ckp_path = os.path.join(flags.ckpdir, "ckp_self_play.tar")
        if flags.ckp: 
            if not os.path.exists(self.ckp_path):
                self.ckp_path = os.path.join(flags.ckpdir, "ckp_actor.tar")
            if not os.path.exists(self.ckp_path):
                raise Exception(f"Cannot find checkpoint in {flags.ckpdir}/ckp_self_play.tar or {flags.ckpdir}/ckp_actor.tar")
            self.load_checkpoint(self.ckp_path)

        self.timer = timeit.default_timer
        self.start_time = self.timer()
        self.sps_buffer = [(self.step, self.start_time)] * 36
        self.sps = 0
        self.sps_buffer_n = 0
        self.sps_start_time, self.sps_start_step = self.start_time, self.step
        self.ckp_start_time = int(time.strftime("%M")) // 10
        

    def insert(self, step_status, episode_return, episode_step, real_done, actor_id):

        stats = {}

        T, B, *_ = episode_return.shape
        last_step_real = (step_status == 0) | (step_status == 3)
        next_step_real = (step_status == 2) | (step_status == 3)

        # extract episode_returns
        if np.any(real_done):            
            episode_returns = episode_return[real_done][
                :, 0
            ]
            episode_returns = tuple(episode_returns)
            episode_lens = episode_step[real_done]
            episode_lens = tuple(episode_lens)
            done_ids = np.broadcast_to(actor_id, real_done.shape)[real_done]
            done_ids = tuple(done_ids)
        else:
            episode_returns, episode_lens, done_ids = (), (), ()

        self.ret_buffers[0].insert(episode_returns, done_ids)
        stats = {"rmean_episode_return": self.ret_buffers[0].get_mean()}

        for prefix in ["im", "cur"]:            
            if prefix == "im":
                done = next_step_real
            elif prefix == "cur":
                done = real_done
            
            if prefix in self.rewards_ls:            
                n = self.rewards_ls.index(prefix)
                self.ret_buffers[n].insert_raw(
                    episode_return,
                    ind=n,
                    actor_id=actor_id,
                    done=done,
                )
                r = self.ret_buffers[n].get_mean()
                stats["rmean_%s_episode_return" % prefix] = r

        self.step += T * B
        self.real_step += np.sum(last_step_real).item()
        self.tot_eps += np.sum(real_done).item()

        stats.update({
            "step": self.step,
            "real_step": self.real_step,
            "tot_eps": self.tot_eps,
            "episode_returns": episode_returns,
            "episode_lens": episode_lens,
            "done_ids": done_ids,
        })

        # write to log file
        self.plogger.log(stats)

        # print statistics
        if self.timer() - self.start_time > 5:
            self.sps_buffer[self.sps_buffer_n] = (self.step, self.timer())

            self.sps_buffer_n = (self.sps_buffer_n + 1) % len(self.sps_buffer)
            self.sps = (
                self.sps_buffer[self.sps_buffer_n - 1][0]
                - self.sps_buffer[self.sps_buffer_n][0]
            ) / (
                self.sps_buffer[self.sps_buffer_n - 1][1]
                - self.sps_buffer[self.sps_buffer_n][1]
            )
            tot_sps = (self.step - self.sps_start_step) / (
                self.timer() - self.sps_start_time
            )
            print_str = (
                "[%s] Steps %i @ %.1f SPS (%.1f). Eps %i. Ret %f (%f/%f)."
                % (
                    self.flags.xpid,
                    self.real_step,
                    self.sps,
                    tot_sps,
                    self.tot_eps,
                    stats["rmean_episode_return"],
                    stats.get("rmean_im_episode_return", 0.),
                    stats.get("rmean_cur_episode_return", 0.),
                )
            )            
            self._logger.info(print_str)
            self.start_time = self.timer()
            self.queue_n = 0     

        if int(time.strftime("%M")) // 10 != self.ckp_start_time:
            self.save_checkpoint()
            self.ckp_start_time = int(time.strftime("%M")) // 10     

        return self.real_step  
    
    def save_checkpoint(self):
        self._logger.info("Saving self-play checkpoint to %s" % self.ckp_path)
        d = {
                "step": self.step,
                "real_step": self.real_step,
                "tot_eps": self.tot_eps,
                "ret_buffers": self.ret_buffers,
                "flags": vars(self.flags),
            }      
        try:
            torch.save(d, self.ckp_path + ".tmp")
            os.replace(self.ckp_path + ".tmp", self.ckp_path)
        except:       
            pass

    def load_checkpoint(self, ckp_path: str):
        train_checkpoint = torch.load(ckp_path, torch.device("cpu"))
        self.step = train_checkpoint["step"]
        self.real_step = train_checkpoint["real_step"]
        self.tot_eps = train_checkpoint["tot_eps"]
        self.ret_buffers = train_checkpoint["ret_buffers"]
        self._logger.info("Loaded self-play checkpoint from %s" % ckp_path)
