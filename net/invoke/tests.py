"""
Tests commands
"""

import invoke

@invoke.task
def unit_tests(context):

    context.run("pytest ./tests", pty=True)


@invoke.task
def static_code_analysis(context):

    directories = "net scripts tests"

    context.run("pycodestyle {}".format(directories))
    context.run("pylint {}".format(directories))
    context.run("xenon . --max-absolute B")


@invoke.task
def commit_stage(context):

    unit_tests(context)
    static_code_analysis(context)
