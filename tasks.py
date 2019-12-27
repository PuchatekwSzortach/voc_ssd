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


tests_collection = invoke.collection.Collection("tests")
tests_collection.add_task(unit_tests)
tests_collection.add_task(static_code_analysis)
tests_collection.add_task(commit_stage)

# Note - name `ns` is important - invoke will recognize only that name as default collection
ns = invoke.Collection()
ns.add_collection(tests_collection)
