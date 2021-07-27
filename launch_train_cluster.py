from itertools import product
from experiment_launcher import Launcher

if __name__ == '__main__':
    local = False
    test = False
    use_cuda = True

    launcher = Launcher(exp_name='cody_finger_b256',
                        python_file='train_cluster',
                        n_exp=5,
                        n_cores=16,
                        memory=1600,
                        days=2,
                        hours=10,
                        minutes=0,
                        seconds=0,
                        n_jobs=1,
                        conda_env='py3.6',
                        gres='gpu:rtx2080:1' if use_cuda else None,
                        use_timestamp=True)

    # curl_lr_list = [1e-5, 1e-6, 1e-7]
    #beta_curl_list = [1e3, 100, 1, 0.1, 0.01, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    #beta_curl_list = [1e3, 1, 0.01, 1e-4, 1e-6, 1e-8]
    #beta_curl_list = [1e5, 1e7]
    #beta_curl_list = [1e6, 1e4, 100]
    launcher.add_default_params(
        domain_name='finger',
        task_name='spin',
        batch_size=256,
        action_repeat=2,
        num_train_steps=251000,
        replay_buffer_capacity=100000,
        eval_freq=5000,
        curl_lr=1e-3,
        omega_curl_loss=100,
        beta_curl=1000
    )

    #for beta_curl in beta_curl_list:
    #    launcher.add_experiment(beta_curl=beta_curl)
    launcher.add_experiment()
    launcher.run(local, test)

