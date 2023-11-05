from q_trainer import q_trainer

if __name__ == "__main__":
    q_trainer = q_trainer(
        state_data_filename='q_table72120231942.pkl',
        PARAM_number_of_episodes=400,
        num_states_in_each_component=10,
        # human_render_mode=True
    )
    q_trainer.run_session()