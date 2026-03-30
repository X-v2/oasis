#![no_std]

use soroban_sdk::{contract, contractimpl, Env, String};

#[contract]
pub struct Contract;

#[contractimpl]
impl Contract {
    // CREATE
    pub fn create(env: Env, analysis_id: String, analysis_hash: String) {
        if env.storage().instance().has(&analysis_id) {
            panic!("Already exists");
        }
        env.storage().instance().set(&analysis_id, &analysis_hash);
    }

    // GET
    pub fn get(env: Env, analysis_id: String) -> String {
        env.storage()
            .instance()
            .get(&analysis_id)
            .unwrap_or_else(|| panic!("Not found"))
    }

    // DELETE
    pub fn delete(env: Env, analysis_id: String) {
        env.storage().instance().remove(&analysis_id);
    }
}