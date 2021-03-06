# -*- coding: utf-8 -*-
"""
Generate the administrator of the neural network, delegated to train/tune the model

NOTE: The login will be made using the `mail` and `password` fields, username is not necessary
"""
import sys

sys.path.insert(0, "../")

from datastructure.Administrator import Administrator

# Creating a new administrator with the following credentials
a = Administrator("username", "mail", "password")
a.init_redis_connection()
print("Remove user -> {}".format(a.remove_user()))
print("Add user -> {}".format(a.add_user()))
print("Verify login {}".format(a.verify_login("password")))
