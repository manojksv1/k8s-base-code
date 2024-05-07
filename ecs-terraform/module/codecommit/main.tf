resource "aws_codecommit_repository" "code_repo" {
  repository_name = var.repo-name
  description     = "reposetroy for code "
}
