"""
Module with invoke tasks
"""

import invoke

import net.invoke.analyze
import net.invoke.host
import net.invoke.train
import net.invoke.tests
import net.invoke.visualize


# Default invoke collection
ns = invoke.Collection()

# Add collections defined in other files
ns.add_collection(net.invoke.analyze)
ns.add_collection(net.invoke.host)
ns.add_collection(net.invoke.train)
ns.add_collection(net.invoke.tests)
ns.add_collection(net.invoke.visualize)
