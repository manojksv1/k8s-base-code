output "alb-dns" {
  value = aws_lb.loadbalancer.dns_name
}