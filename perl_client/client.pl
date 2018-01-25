#!/usr/bin/env perl

use HTTP::Request;
use LWP::UserAgent;

main();

sub main {
  my $endpoint = "http://127.0.0.1:8500";
  my $json = '{"keys": [[11.0], [2.0]], "features": [[1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]]}';
  my $req = HTTP::Request->new( 'POST', $endpoint );
  $req->header( 'Content-Type' => 'application/json' );
  $req->content( $json );
  $ua = LWP::UserAgent->new;

  $response = $ua->request($req);
  print($response->content)
}


