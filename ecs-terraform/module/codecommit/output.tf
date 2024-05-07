output "reposetroy_uri" {
  value = aws_codecommit_repository.code_repo.clone_url_http
}