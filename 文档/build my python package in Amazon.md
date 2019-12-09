[TOC]
# 1.0 Create version set
Version sets can be created in two ways: via the web GUI at code.amazon.com, or via the command line.

Version sets can be owned by a posix group, and LDAP group, or a bindle. Bindles are the recommended approach. Learn more about bindles on the wiki. Obtain your personal bind ID from bindles.amazon.com. Click on "My Personal Bindle" and note your bindle ID. Your bindle ID will be of the form amzn1.bindle.resource.string_of_random_looking_characters.

Create your version set with the command below. Since Redhat Linux (platforms RHEL5 and RHEL5_64) are being deprecated, be sure to create any new package for the AL2012 platform. 

    brazil vs create -vs DCCD-ctpn-yuankuc/dev --platforms AL2012 --bindleID amzn1.bindle.resource.35vyrhd2ohnto6cdovrk4t73q --from live --mailingList yuankuc@amazon.com

In order for your package to become the target of a version set, we need to ensure that Brazil knows about your package and has a copy of your code. You will use brazil-octane to promote and push your package into a Brazil package to the Amazon code repository.

    cd src/PythonHelloWorld-yuankuc
    brazil-octane package promote --bindleName="yuankuc's PersonalSoftwareBindle" --push

## Build your version set
Now that you have a Brazil package and a version set, we can build the version set.

1. Visit build.amazon.com.
2. In the Version Set search box, enter PythonHelloWorld-yuankuc/dev. Select your version set by clicking on it when it appears.
1. Enter your package name, PythonHelloWorld-yuankuc in the Package name search field, and click to accept.
4. In the pop-up that appears, click select next to the most recent commit.
5. Click the checkbox for Mark as target. Unable to import a version set with no targets
6. Click the checkbox for Auto-merge Dependencies.
7. Finally, click the button Submit Build Request.
   
When the build succeeds, configure your workspace to use the new version set.

    brazil ws use -vs PythonHelloWorld-yuankuc/dev

# 1.1 Set up your workspace

## 1.1.1 Set up a directory to contain your workspaces

Each package that you work on will be contained in a workspace. It's convenient to keep these workspaces in a subdirectory of your home directory on your cloud desktop. Create a directory to hold your workspaces:

    mkdir -p ~/workspaces && cd ~/workspaces

## 1.1.2 Create a new workspace
Create a new workspace called "PythonHelloWorld" using brazil. Make sure your package name start with a capital letter, otherwise, you may encounter some error.

    brazil ws create --name PythonHelloWorld
    cd PythonHelloWorld

## 1.1.3 Populate the workspace with a templated BrazilPython3 package
The brazil-octane command can create all the scaffolding needed for an Amazon package for you. To use the Python template, which will create an empty BrazilPython3 package from a template.

NB: By default, the pkg-python template will create a workspace with CPython 2.7 and 1.4. This may change; check the documentation. You can specify which interpreter(s) will be built in your package by editing ```/build-tools/bin/python-builds```

```
brazil-octane package generate pkg-python --name PythonHelloWorld
```
Expect a lot of output ending with something like the following.

    package PythonHelloWorld created successfully
    [2017-10-18 17:42:20.858986]:
    Please note that the generated package(s) are local workspace only package(s).
For getting more help on converting them in to real Brazil package, type:
    
    brazil-octane help pkg promote

Move into the package's root directory

    cd src/PythonHelloWorld

Specify a version set.

    brazil ws use -vs <VersionSetName>

# 1.2 Create your "Hello, world" script
Before we start changing files in our package, it's a good idea to start a new git branch. We'll start a new branch called dev with the following command.

    git checkout -b dev
Now we're ready to start editing files.

Create a Python module. The Octane template has already set up a directory for us. Begin editing the file with your favorite editor:

    vim src/python_hello_world/hello_world.py

Next, edit your Config file and add the following to the dependencies section. This dependency is required for executable, auto-created bin wrapper scripts that you can use to run the code in your module.
    
    Python = default;
    Python-setuptools = default;

Everytime you update Config, re-sync your workspace:

    brazil ws sync --md

You also need to tell Brazil where the entry point is in your module. Edit setup.py and place the following instructions into the setup() function:

     entry_points="""\
                  [console_scripts]
                  PythonHelloWorld = python_hello_world.hello_world:main
                  """,