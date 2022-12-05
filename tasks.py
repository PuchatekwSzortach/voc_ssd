"""
Module with invoke tasks
"""

import invoke

import net.invoke.host
import net.invoke.tests


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(net.invoke.host)
ns.add_collection(net.invoke.tests)
