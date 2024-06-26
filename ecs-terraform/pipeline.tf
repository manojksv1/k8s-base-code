# resource "aws_codepipeline" "python_app_pipeline" {
#   name     = "python-app-pipeline"
#   role_arn = aws_iam_role.apps_codepipeline_role.arn
#   tags = {
#     Environment = var.env
#   }

#   artifact_store {
#     location = module.artifacts-bucket.cicd-state-bucket-name
#     type     = "S3"

#   }

#   stage {
#     name = "Source"

#     action {
#       category = "Source"
#       configuration = {
#         "BranchName" = "main"
#         # "PollForSourceChanges" = "false"
#         "RepositoryName" = var.repo-name
#       }
#       input_artifacts = []
#       name            = "Source"
#       output_artifacts = [
#         "SourceArtifact",
#       ]
#       owner     = "AWS"
#       provider  = "CodeCommit"
#       run_order = 1
#       version   = "1"
#     }
#   }
#   stage {
#     name = "Build"

#     action {
#       category = "Build"
#       configuration = {
#         "EnvironmentVariables" = jsonencode(
#           [
#             {
#               name  = "environment"
#               type  = "PLAINTEXT"
#               value = var.env
#             },
#             {
#               name  = "AWS_DEFAULT_REGION"
#               type  = "PLAINTEXT"
#               value = var.aws-region
#             },
#             {
#               name  = "AWS_ACCOUNT_ID"
#               type  = "PARAMETER_STORE"
#               value = "637423386220"
#             },
#             {
#               name  = "IMAGE_REPO_NAME"
#               type  = "PLAINTEXT"
#               value = module.ecr.repo-name-ecr
#             },
#             {
#               name  = "IMAGE_TAG"
#               type  = "PLAINTEXT"
#               value = "latest"
#             },
#             {
#               name  = "CONTAINER_NAME"
#               type  = "PLAINTEXT"
#               value = "web-page"
#             },
#           ]
#         )
#         "ProjectName" = aws_codebuild_project.containerAppBuild.name
#       }
#       input_artifacts = [
#         "SourceArtifact",
#       ]
#       name = "Build"
#       output_artifacts = [
#         "BuildArtifact",
#       ]
#       owner     = "AWS"
#       provider  = "CodeBuild"
#       run_order = 1
#       version   = "1"
#     }
#   }
#   stage {
#     name = "Deploy"

#     action {
#       category = "Deploy"
#       configuration = {
#         "ClusterName" = module.ecs.ecs-name
#         "ServiceName" = module.ecs.ServiceName.name
#         "FileName"    = "imagedefinitions.json"
#         #"DeploymentTimeout" = "15"
#       }
#       input_artifacts = [
#         "BuildArtifact",
#       ]
#       name             = "Deploy"
#       output_artifacts = []
#       owner            = "AWS"
#       provider         = "ECS"
#       run_order        = 1
#       version          = "1"
#     }
#   }

#   # depends_on = [
#   #   aws_codebuild_project.containerAppBuild,
#   #   aws_ecs_cluster.python_app_cluster,
#   #   aws_ecs_service.python_service,
#   #   aws_ecr_repository.python_app_repo,
#   #   aws_codecommit_repository.code_repo,
#   #   aws_s3_bucket.cicd_bucket,
#   # ]
# }